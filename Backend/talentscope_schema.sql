-- 1. Resumes table: stores raw & cleaned text
CREATE TABLE resumes (
    id SERIAL PRIMARY KEY,
    name TEXT,
    email TEXT,
    phone TEXT,
    raw_text TEXT NOT NULL,
    cleaned_text TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 2. Structured JSON from LLM Pass #1
CREATE TABLE resume_structured (
    resume_id INT REFERENCES resumes(id) ON DELETE CASCADE,
    structured_json JSONB NOT NULL,
    PRIMARY KEY (resume_id)
);

-- 3. Trait scores from LLM Pass #2
CREATE TABLE resume_traits (
    resume_id INT REFERENCES resumes(id) ON DELETE CASCADE,
    leadership FLOAT,
    communication FLOAT,
    analytical_thinking FLOAT,
    ownership FLOAT,
    problem_solving FLOAT,
    attention_to_detail FLOAT,
    PRIMARY KEY (resume_id)
);

-- 4. Optional: mapping table for FAISS → SQL (later for sync)
CREATE TABLE resume_faiss_map (
    resume_id INT REFERENCES resumes(id) ON DELETE CASCADE,
    faiss_index INT UNIQUE,
    PRIMARY KEY (resume_id)
);

CREATE TABLE jobs (
    job_id SERIAL PRIMARY KEY,
    title TEXT,
    description TEXT,
    skills TEXT[]
);