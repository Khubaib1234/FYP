import requests
import json
import logging

url = "http://127.0.0.1:8000/upload-resume/"

# Example mock payload from user's issue
structured_json = {
    "name": "Khubaib Ahmed Jamil",
    "email": "khubaibahmedjamil@gmail.com",
    "phone": "0334-2212963",
    "education": [
        {
            "degree": "Bachelors in Computer Science",
            "institution": "FAST National University of Computer and Emerging Sciences",
            "start_year": 2022,
            "end_year": None
        }
    ],
    "experience": [],
    "skills": ["C", "Assembly", "Python", "HTML", "CSS", "MySQL", "OracleSQL", "MongoDB", "Machine Learning"]
}

try:
    with open("dummy.txt", "w") as f:
        f.write(json.dumps(structured_json))
    with open("dummy.txt", "rb") as f:
        response = requests.post(url, files={"file": f})
    print(f"Status: {response.status_code}")
    print(response.text)
except Exception as e:
    print("Error:", e)
