import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------
# Directories
# -------------------------
CLEANED_JSON_DIR = "JSON"          # LLM Pass 1 output
TRAITS_JSON_DIR = "Traits_JSON"    # LLM Pass 2 output
FAISS_INDEX_DIR = "FAISS_Index"    # Store FAISS index and candidate mapping
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# -------------------------
# Load embedding model
# -------------------------
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
embedding_dim = model.get_sentence_embedding_dimension()

# -------------------------
# Flatten JSON for embedding
# -------------------------
def flatten_resume_json(resume_json: dict) -> str:
    parts = []
    parts.append(resume_json.get('name', ''))
    parts.append(resume_json.get('email', ''))
    parts.append(resume_json.get('phone', ''))

    for edu in resume_json.get('education', []):
        parts.append(f"{edu.get('degree','')} {edu.get('institution','')} {edu.get('start_year','')} {edu.get('end_year','')}")

    for exp in resume_json.get('experience', []):
        parts.append(f"{exp.get('job_title','')} at {exp.get('company','')} {exp.get('start_year','')}-{exp.get('end_year','')}. {exp.get('description','')}")

    projects = []
    for p in resume_json.get('projects', []):
        if isinstance(p, dict):
            projects.append(p.get('title', '') + ' ' + p.get('description', ''))
        else:
            projects.append(str(p))
    parts.append('Projects: ' + ', '.join(projects))

    parts.append('Skills: ' + ', '.join([str(s) for s in resume_json.get('skills', [])]))
    parts.append('Certifications: ' + ', '.join([str(c) for c in resume_json.get('certifications', [])]))

    return ' '.join(parts)

def flatten_traits_json(traits_json: dict) -> str:
    return ' '.join([f"{k}: {v}" for k, v in traits_json.items()])

# -------------------------
# Add a single candidate to FAISS index
# -------------------------
def add_candidate_to_index(resume_json_path, traits_json_path, candidate_name, index_path=None, candidate_ids_path=None):
    with open(resume_json_path, 'r', encoding='utf-8') as f:
        resume_json = json.load(f)
    resume_text = flatten_resume_json(resume_json)
    resume_embedding = model.encode(resume_text).astype('float32')

    if index_path and os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatL2(len(resume_embedding))

    if candidate_ids_path and os.path.exists(candidate_ids_path):
        with open(candidate_ids_path, 'r', encoding='utf-8') as f:
            candidate_ids = json.load(f)
    else:
        candidate_ids = []

    index.add(np.expand_dims(resume_embedding, axis=0))
    candidate_ids.append(candidate_name)

    if index_path:
        faiss.write_index(index, index_path)
    if candidate_ids_path:
        with open(candidate_ids_path, 'w', encoding='utf-8') as f:
            json.dump(candidate_ids, f)

    return index, candidate_ids
