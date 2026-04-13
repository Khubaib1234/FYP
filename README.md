# TalentScope AI - Run Guide

## Prerequisites
1.  **Python 3.8+** installed.
2.  **Neo4j Database** running locally.
    -   URI: `bolt://localhost:7687`
    -   User: `neo4j`
    -   Password: `saadbrohi` (or initialize environment variables `NEO4J_USER` and `NEO4J_PASSWORD`)

## 1. Backend Setup

Open a terminal in the `Backend` directory:

```bash
cd Backend
pip install -r requirements.txt
python main.py
```

The backend will start at `http://127.0.0.1:8000`.

## 2. Frontend Setup

Open a **new** terminal in the `frontend` directory:

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

The frontend will open in your browser.
