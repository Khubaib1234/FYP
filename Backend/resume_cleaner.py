import os
import re
import pdfplumber
from docx import Document

# -------------------------
# Cleaning utilities
# -------------------------
def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def strip_noise(text: str) -> str:
    text = re.sub(r'[^A-Za-z0-9\s.,:;()\-@/]', ' ', text)
    return normalize_whitespace(text)

def read_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def read_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def clean_resume_file(input_file: str) -> str:
    ext = os.path.splitext(input_file)[1].lower()
    if ext == ".txt":
        raw_text = read_txt(input_file)
    elif ext == ".pdf":
        raw_text = read_pdf(input_file)
    elif ext == ".docx":
        raw_text = read_docx(input_file)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return strip_noise(raw_text)

def save_cleaned_text(cleaned_text: str, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
