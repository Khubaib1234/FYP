import os
import shutil
import json
import numpy as np
import faiss
import logging
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from resume_cleaner import clean_resume_file, save_cleaned_text
from llm_processor import extract_structured_json, save_json_output
from llm_pass_2 import infer_traits, save_traits_json
from embedding import model, flatten_resume_json, flatten_traits_json
from db import get_connection, release_connection
from graph_builder import insert_candidate_graph

# -------------------------
# Setup directories
# -------------------------
UPLOAD_DIR = "Uploads"
CLEANED_DIR = "Cleaned"
JSON_DIR = "JSON"
TRAITS_DIR = "Traits_JSON"
FAISS_INDEX_DIR = "FAISS_Index"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(TRAITS_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# -------------------------
# FAISS index paths
# -------------------------
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, 'candidate_index.faiss')
CANDIDATE_IDS_PATH = os.path.join(FAISS_INDEX_DIR, 'candidate_ids.json')
SNIPPET_METADATA_PATH = os.path.join(FAISS_INDEX_DIR, 'snippet_metadata.json')

# Load or create FAISS index
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CANDIDATE_IDS_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CANDIDATE_IDS_PATH, 'r', encoding='utf-8') as f:
        candidate_ids = json.load(f)
else:
    index = None
    candidate_ids = []

if os.path.exists(SNIPPET_METADATA_PATH):
    with open(SNIPPET_METADATA_PATH, 'r', encoding='utf-8') as f:
        snippet_metadata = json.load(f)
else:
    snippet_metadata = {}

# -------------------------
# FastAPI Router
# -------------------------
router = APIRouter()

@router.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    try:
        name = os.path.splitext(file.filename)[0]

        # 1. Save uploaded file
        upload_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Clean resume
        cleaned_text = clean_resume_file(upload_path)
        cleaned_path = os.path.join(CLEANED_DIR, f"{name}_cleaned.txt")
        save_cleaned_text(cleaned_text, cleaned_path)

        # 3. LLM Pass 1: Structured JSON
        structured_json = extract_structured_json(cleaned_text)
        json_path = os.path.join(JSON_DIR, f"{name}_structured.json")
        save_json_output(structured_json, json_path)

        # 4. LLM Pass 2: Trait Extraction
        trait_json = infer_traits(cleaned_text)
        traits_path = os.path.join(TRAITS_DIR, f"{name}_traits.json")
        save_traits_json(trait_json, traits_path)

        # 5. Embeddings + Save to PostgreSQL
        resume_text = flatten_resume_json(structured_json)
        # We only use resume_text for FAISS to match the retriever's search queries
        resume_embedding = model.encode(resume_text).astype('float32')

        # -------------------------
        # Save to PostgreSQL
        # -------------------------
        conn = get_connection()
        cur = conn.cursor()

        # Insert into resumes table
        cur.execute("""
            INSERT INTO resumes (name, email, phone, raw_text, cleaned_text)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (
            structured_json.get('name'),
            structured_json.get('email'),
            structured_json.get('phone'),
            cleaned_text,
            cleaned_text
        ))
        resume_id = cur.fetchone()[0]

        # Insert structured JSON
        cur.execute("""
            INSERT INTO resume_structured (resume_id, structured_json)
            VALUES (%s, %s)
        """, (resume_id, json.dumps(structured_json)))

        # Insert trait scores
        cur.execute("""
            INSERT INTO resume_traits (resume_id, leadership, communication,
                analytical_thinking, ownership, problem_solving, attention_to_detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            resume_id,
            trait_json.get('leadership'),
            trait_json.get('communication'),
            trait_json.get('analytical_thinking'),
            trait_json.get('ownership'),
            trait_json.get('problem_solving'),
            trait_json.get('attention_to_detail')
        ))

        # -------------------------
        # Add to FAISS and metadata
        # -------------------------
        global index, candidate_ids, snippet_metadata

        # Initialize FAISS index if first time
        if index is None:
            index = faiss.IndexFlatL2(len(resume_embedding))

        # Add new embedding
        index.add(np.expand_dims(resume_embedding, axis=0))
        faiss_idx = index.ntotal - 1
        candidate_ids.append(resume_id)

        # Update snippet metadata
        snippet_metadata[str(faiss_idx)] = {
            "candidate_id": str(resume_id),
            "text": resume_text
        }

        # Save updated index + IDs + metadata
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(CANDIDATE_IDS_PATH, 'w', encoding='utf-8') as f:
            json.dump(candidate_ids, f)
        with open(SNIPPET_METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(snippet_metadata, f, indent=4)

        # FAISS → SQL Sync
        cur.execute("""
            INSERT INTO resume_faiss_map (resume_id, faiss_index)
            VALUES (%s, %s)
            ON CONFLICT (resume_id) DO NOTHING
        """, (resume_id, faiss_idx))

        conn.commit()
        cur.close()
        release_connection(conn)

        # Neo4j Graph Sync
        try:
            insert_candidate_graph(
                resume_id=resume_id,
                structured_json=structured_json,
                traits=trait_json
            )
        except Exception as neo_err:
            print("Neo4j Sync Error:", neo_err)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Resume fully processed successfully",
                "uploaded_file": upload_path,
                "cleaned_file": cleaned_path,
                "structured_json_file": json_path,
                "traits_json_file": traits_path,
                "structured_output": structured_json,
                "traits_output": trait_json
            }
        )

    except Exception as e:
        import traceback
        logging.error("Error in upload_resume: %s", str(e))
        logging.error(traceback.format_exc())
        if 'conn' in locals():
            conn.rollback()
            release_connection(conn)
        return JSONResponse(status_code=500, content={"error": str(e)})
