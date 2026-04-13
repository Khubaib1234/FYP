from fastapi import APIRouter
from fastapi.responses import JSONResponse
from db import get_connection, release_connection
import json

router = APIRouter()

@router.get("/candidates")
async def get_all_candidates():
    """
    Fetch all candidates' structured JSON + traits from the database.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Fetch all candidate IDs + structured JSON
        cur.execute("""
            SELECT r.id, rs.structured_json, rt.leadership, rt.communication,
                   rt.analytical_thinking, rt.ownership, rt.problem_solving, rt.attention_to_detail
            FROM resumes r
            JOIN resume_structured rs ON r.id = rs.resume_id
            LEFT JOIN resume_traits rt ON r.id = rt.resume_id
        """)
        rows = cur.fetchall()

        candidates = []
        for row in rows:
            candidate_id = row[0]
            structured_json = row[1]
            traits_json = {
                "leadership": row[2],
                "communication": row[3],
                "analytical_thinking": row[4],
                "ownership": row[5],
                "problem_solving": row[6],
                "attention_to_detail": row[7]
            }
            candidates.append({
                "candidate_id": candidate_id,
                "structured_json": structured_json,
                "traits": traits_json
            })

        cur.close()
        release_connection(conn)

        return JSONResponse(status_code=200, content=candidates)

    except Exception as e:
        if 'conn' in locals():
            release_connection(conn)
        return JSONResponse(status_code=500, content={"error": str(e)})
