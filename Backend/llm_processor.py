import os
import json
import logging
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq

# -------------------------
# Setup logging
# -------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------
# Directories and files
# -------------------------
CLEANED_DIR = "Cleaned"
JSON_DIR = "JSON"
PROMPT_FILE = "prompt_templates/resume_extraction.txt"

os.makedirs(JSON_DIR, exist_ok=True)

# -------------------------
# Load prompt template
# -------------------------
def load_prompt_template(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------
# Helper functions
# -------------------------
def _parse_response_text(response_text: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    response_text = re.sub(r'^```json|```$', '', response_text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        logging.warning("Failed to parse LLM output as JSON.")
        return {}

def _empty_resume_structure() -> dict:
    return {
        "name": "",
        "email": "",
        "phone": "",
        "education": [],
        "experience": [],
        "skills": [],
        "projects": [],
        "certifications": []
    }

# -------------------------
# Extract structured JSON from resume text
# -------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"

def extract_structured_json(cleaned_text: str, llm_model: str = "qwen2.5:7b-instruct-q5_K_M") -> dict:
    prompt_text = load_prompt_template(PROMPT_FILE)
    prompt_text = prompt_text.replace("{resume_text}", cleaned_text)

    # ── Try Ollama first ──────────────────────────────────────────────────────
    try:
        logging.info("Attempting Ollama model: %s", llm_model)
        llm = OllamaLLM(model=llm_model, temperature=0, timeout=30)
        response = llm.generate([prompt_text])
        response_text = response.generations[0][0].text
        logging.info("Ollama response received (first 500 chars): %s", response_text[:500])
        parsed = _parse_response_text(response_text)
        if parsed:
            return parsed
    except Exception as e:
        logging.warning("Ollama unavailable or failed (%s). Falling back to Groq API.", str(e))

    # ── Fallback: Groq API ────────────────────────────────────────────────────
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logging.error("GROQ_API_KEY environment variable is not set. Cannot use Groq fallback.")
        return _empty_resume_structure()

    try:
        logging.info("Calling Groq API with model: %s", GROQ_MODEL)
        groq_llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=groq_api_key)
        groq_response = groq_llm.invoke(prompt_text)
        response_text = groq_response.content
        logging.info("Groq response received (first 500 chars): %s", response_text[:500])
        return _parse_response_text(response_text) or _empty_resume_structure()

    except Exception as e:
        logging.error("Groq API call failed: %s", str(e))

    return _empty_resume_structure()


# -------------------------
# Save JSON output
# -------------------------
def save_json_output(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
