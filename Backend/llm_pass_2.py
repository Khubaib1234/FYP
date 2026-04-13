import os
import json
import logging
import re
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# -------------------------
# Setup logging
# -------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------
# Directories and files
# -------------------------
TRAITS_JSON_DIR = "Traits_JSON"
PROMPT_FILE = "prompt_templates/traits_extraction.txt"
os.makedirs(TRAITS_JSON_DIR, exist_ok=True)

# -------------------------
# Load prompt template
# -------------------------
def load_prompt_template(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------
# Infer traits from resume text
# -------------------------
# Groq model to use when Ollama is unavailable
GROQ_MODEL = "llama-3.3-70b-versatile"

def _parse_response_text(response_text: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    response_text = re.sub(r'^```json|```$', '', response_text.strip(), flags=re.MULTILINE).strip()
    return json.loads(response_text)

def _empty_traits() -> dict:
    return {
        "leadership": 0.0,
        "communication": 0.0,
        "analytical_thinking": 0.0,
        "ownership": 0.0,
        "problem_solving": 0.0,
        "attention_to_detail": 0.0
    }

def infer_traits(clean_text: str, llm_model: str = "qwen2.5:7b-instruct-q5_K_M") -> dict:
    prompt_text = load_prompt_template(PROMPT_FILE)
    prompt_text = prompt_text.replace("{resume}", clean_text)

    # ── Try Ollama first ──────────────────────────────────────────────────────
    try:
        logging.info("Attempting Ollama model: %s", llm_model)
        llm = OllamaLLM(model=llm_model, temperature=0)
        response = llm.generate([prompt_text])
        response_text = response.generations[0][0].text
        logging.info("Ollama response received (first 500 chars): %s", response_text[:500])
        return _parse_response_text(response_text)

    except json.JSONDecodeError:
        logging.warning("Ollama output was not valid JSON, trying Groq fallback.")
    except Exception as e:
        logging.warning("Ollama unavailable (%s). Falling back to Groq API.", str(e))

    # ── Fallback: Groq API ────────────────────────────────────────────────────
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logging.error("GROQ_API_KEY environment variable is not set. Cannot use Groq fallback.")
        return _empty_traits()

    try:
        logging.info("Calling Groq API with model: %s", GROQ_MODEL)
        groq_llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=groq_api_key)
        groq_response = groq_llm.invoke(prompt_text)
        response_text = groq_response.content
        logging.info("Groq response received (first 500 chars): %s", response_text[:500])
        return _parse_response_text(response_text)

    except json.JSONDecodeError:
        logging.warning("Groq output was not valid JSON, returning empty trait structure.")
    except Exception as e:
        logging.error("Groq API call failed: %s", str(e))

    return _empty_traits()

# -------------------------
# Save traits JSON
# -------------------------
# Dynamically create any subfolders in Traits_JSON if included in filename
def save_traits_json(trait_json: dict, filename: str):
    path = os.path.join(TRAITS_JSON_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trait_json, f, indent=4)