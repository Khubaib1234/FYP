import requests
import os
import glob
import traceback

url = "http://127.0.0.1:8000/upload-resume/"
pdf_path = r"e:\fyp\FYP\Dataset\resume-v2.pdf"  # Try finding user's specific PDF

if not os.path.exists(pdf_path):
    files = glob.glob(r"e:\fyp\FYP\Dataset\*\*.pdf")
    if files:
        pdf_path = files[0]

if os.path.exists(pdf_path):
    try:
        print(f"Uploading {pdf_path}...")
        with open(pdf_path, "rb") as f:
            response = requests.post(url, files={"file": f})
        print(f"Status: {response.status_code}")
        if response.status_code == 500:
            print("ERROR DETAILS:")
            print(response.text)
        else:
            print(response.json())
    except:
        traceback.print_exc()
else:
    print("Could not find any PDF to upload.")
