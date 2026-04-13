# graph_builder.py
from neo4j_client import neo4j_client

def insert_candidate_graph(resume_id, structured_json, traits):

    # --- Candidate ---
    name = structured_json.get("name", "Unknown")
    neo4j_client.run("""
        MERGE (c:Candidate {id: $id})
        SET c.name = $name
    """, {"id": str(resume_id), "name": name})

    # --- Skills ---
    raw_skills = structured_json.get("skills", [])
    skills_list = []

    for entry in raw_skills:
        # Check if entry is a comma-separated string
        if isinstance(entry, str):
            # Remove category prefix if present (e.g., "Programming: Python, C")
            if ":" in entry:
                entry = entry.split(":", 1)[1]
            # Split by commas
            split_skills = [s.strip() for s in entry.split(",") if s.strip()]
            skills_list.extend(split_skills)
        else:
            # If already a list, just add
            skills_list.append(entry)

    # Insert skills into Neo4j
    for skill in skills_list:
        neo4j_client.run("""
            MERGE (s:Skill {name: $skill})
            MERGE (c:Candidate {id: $id})-[:HAS_SKILL]->(s)
        """, {"skill": skill, "id": str(resume_id)})

    # --- Experience ---
    for exp in structured_json.get("experience", []):
        company = exp.get("company")
        role = exp.get("job_title", "")  # corrected key from 'role' to 'job_title'

        if not company:
            continue

        neo4j_client.run("""
            MERGE (cmp:Company {name: $company})
            MERGE (r:Role {name: $role})
            MERGE (c:Candidate {id: $id})-[:WORKED_AT]->(cmp)
            MERGE (c)-[:PERFORMED_ROLE]->(r)
        """, {
            "company": company,
            "role": role,
            "id": str(resume_id)
        })

    # --- Education ---
    for edu in structured_json.get("education", []):
        inst = edu.get("institution")
        degree = edu.get("degree", "")

        if not inst:
            continue

        neo4j_client.run("""
            MERGE (e:Education {institution: $inst, degree: $degree})
            MERGE (c:Candidate {id: $id})-[:HAS_EDUCATION]->(e)
        """, {
            "inst": inst,
            "degree": degree,
            "id": str(resume_id)
        })

    # --- Traits ---
    for trait_name, score in traits.items():
        neo4j_client.run("""
            MERGE (t:Trait {name: $trait})
            MERGE (c:Candidate {id: $id})-[:EXHIBITS {score: $score}]->(t)
        """, {
            "trait": trait_name,
            "score": float(score),
            "id": str(resume_id)
        })
