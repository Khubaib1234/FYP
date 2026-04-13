import requests
import os
import glob

# 1. Upload a resume
files = glob.glob(r"e:\fyp\FYP\Dataset\*.pdf")
if not files:
    files = glob.glob(r"e:\fyp\FYP\Dataset\*\*.pdf")

if files:
    test_pdf = files[0]
    print(f"Uploading {test_pdf}...")
    with open(test_pdf, "rb") as f:
        response = requests.post("http://127.0.0.1:8000/upload-resume/", files={"file": f})
    print("Upload response:", response.status_code)
    try:
        print(response.json())
    except:
        print(response.text)
else:
    print("No PDFs found in Dataset to upload.")

# 2. Test search
print("\nTesting Search...")
search_payload = {"query": "Machine Learning Engineer with Python skills", "top_k": 3}
response = requests.post("http://127.0.0.1:8000/search", json=search_payload)
print("Search response:", response.status_code)
try:
    print(response.json())
except:
    print(response.text)
