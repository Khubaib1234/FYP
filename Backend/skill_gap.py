from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from db import get_connection, release_connection

router = APIRouter()


# -------------------------
# Pydantic Models
# -------------------------
class JobCreate(BaseModel):
    title: str
    description: Optional[str] = ""
    skills: List[str]


# -------------------------
# GET /jobs - List all jobs
# -------------------------
@router.get("/jobs")
async def get_all_jobs():
    """Return all jobs with their required skills."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT job_id, title, description, skills FROM jobs ORDER BY job_id")
        rows = cur.fetchall()
        cur.close()
        release_connection(conn)

        jobs = [
            {
                "job_id": row[0],
                "title": row[1],
                "description": row[2],
                "skills": row[3] if row[3] else []
            }
            for row in rows
        ]
        return JSONResponse(status_code=200, content=jobs)

    except Exception as e:
        if 'conn' in locals():
            release_connection(conn)
        return JSONResponse(status_code=500, content={"error": str(e)})


# -------------------------
# POST /jobs - Create a new job
# -------------------------
@router.post("/jobs")
async def create_job(job: JobCreate):
    """Create a new job posting with required skills."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO jobs (title, description, skills) VALUES (%s, %s, %s) RETURNING job_id",
            (job.title, job.description, job.skills)
        )
        job_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        release_connection(conn)

        return JSONResponse(
            status_code=201,
            content={
                "message": "Job created successfully",
                "job_id": job_id,
                "title": job.title,
                "skills": job.skills
            }
        )

    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
            release_connection(conn)
        return JSONResponse(status_code=500, content={"error": str(e)})


# -------------------------
# Helper: normalize skills list
# -------------------------
def _normalize_skills(skills) -> set:
    """Lowercase and strip all skills for case-insensitive comparison."""
    if not skills:
        return set()
    return {s.strip().lower() for s in skills if s and s.strip()}


# -------------------------
# GET /skill-gap/{job_id}/{candidate_id}
# -------------------------
@router.get("/skill-gap/{job_id}/{candidate_id}")
async def get_skill_gap(job_id: int, candidate_id: int):
    """
    Compare the skills required by a job against a candidate's skills.
    Returns matched skills, missing skills, and a match percentage.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Fetch job skills
        cur.execute("SELECT title, skills FROM jobs WHERE job_id = %s", (job_id,))
        job_row = cur.fetchone()
        if not job_row:
            cur.close()
            release_connection(conn)
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job_title = job_row[0]
        job_skills = _normalize_skills(job_row[1])

        # Fetch candidate skills from structured_json
        cur.execute(
            """
            SELECT r.id, rs.structured_json
            FROM resumes r
            JOIN resume_structured rs ON r.id = rs.resume_id
            WHERE r.id = %s
            """,
            (candidate_id,)
        )
        candidate_row = cur.fetchone()
        if not candidate_row:
            cur.close()
            release_connection(conn)
            raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found")

        structured_json = candidate_row[1]
        candidate_name = structured_json.get("name", f"Candidate {candidate_id}")
        raw_candidate_skills = structured_json.get("skills", [])
        candidate_skills = _normalize_skills(raw_candidate_skills)

        cur.close()
        release_connection(conn)

        # Compute gap
        matched = sorted(job_skills & candidate_skills)
        missing = sorted(job_skills - candidate_skills)
        total_required = len(job_skills)
        match_pct = round((len(matched) / total_required * 100) if total_required > 0 else 0.0, 1)

        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "job_title": job_title,
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "total_required_skills": total_required,
                "matched_skills": matched,
                "missing_skills": missing,
                "matched_count": len(matched),
                "missing_count": len(missing),
                "match_percentage": match_pct
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        if 'conn' in locals():
            release_connection(conn)
        return JSONResponse(status_code=500, content={"error": str(e)})


# -------------------------
# GET /rank-candidates/{job_id}
# -------------------------
@router.get("/rank-candidates/{job_id}")
async def rank_candidates(job_id: int):
    """
    Rank all candidates by their skill match percentage for the given job.
    Returns candidates sorted from highest to lowest match.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Fetch job
        cur.execute("SELECT title, skills FROM jobs WHERE job_id = %s", (job_id,))
        job_row = cur.fetchone()
        if not job_row:
            cur.close()
            release_connection(conn)
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job_title = job_row[0]
        job_skills = _normalize_skills(job_row[1])
        total_required = len(job_skills)

        # Fetch all candidates
        cur.execute(
            """
            SELECT r.id, rs.structured_json
            FROM resumes r
            JOIN resume_structured rs ON r.id = rs.resume_id
            ORDER BY r.id
            """
        )
        rows = cur.fetchall()
        cur.close()
        release_connection(conn)

        ranked = []
        for row in rows:
            candidate_id = row[0]
            structured_json = row[1]
            candidate_name = structured_json.get("name", f"Candidate {candidate_id}")
            raw_skills = structured_json.get("skills", [])
            candidate_skills = _normalize_skills(raw_skills)

            matched = sorted(job_skills & candidate_skills)
            missing = sorted(job_skills - candidate_skills)
            match_pct = round((len(matched) / total_required * 100) if total_required > 0 else 0.0, 1)

            ranked.append({
                "rank": 0,  # filled in below
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "matched_skills": matched,
                "missing_skills": missing,
                "matched_count": len(matched),
                "missing_count": len(missing),
                "match_percentage": match_pct
            })

        # Sort by match_percentage descending, then by candidate_id ascending as tiebreaker
        ranked.sort(key=lambda x: (-x["match_percentage"], x["candidate_id"]))
        for i, item in enumerate(ranked):
            item["rank"] = i + 1

        return JSONResponse(
            status_code=200,
            content={
                "job_id": job_id,
                "job_title": job_title,
                "total_required_skills": total_required,
                "candidates": ranked
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        if 'conn' in locals():
            release_connection(conn)
        return JSONResponse(status_code=500, content={"error": str(e)})


# -------------------------
# GET /recommend-jobs/{candidate_id}
# -------------------------
@router.get("/recommend-jobs/{candidate_id}")
async def recommend_jobs(candidate_id: int):
    """
    Rank all jobs by how well they match a specific candidate's skills.
    Returns jobs sorted from best match to worst.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Fetch candidate skills from structured_json
        cur.execute(
            """
            SELECT r.id, rs.structured_json
            FROM resumes r
            JOIN resume_structured rs ON r.id = rs.resume_id
            WHERE r.id = %s
            """,
            (candidate_id,)
        )
        candidate_row = cur.fetchone()
        if not candidate_row:
            cur.close()
            release_connection(conn)
            raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found")

        structured_json = candidate_row[1]
        candidate_name = structured_json.get("name", f"Candidate {candidate_id}")
        raw_skills = structured_json.get("skills", [])
        candidate_skills = _normalize_skills(raw_skills)

        # Fetch all jobs
        cur.execute("SELECT job_id, title, description, skills FROM jobs ORDER BY job_id")
        job_rows = cur.fetchall()
        cur.close()
        release_connection(conn)

        if not job_rows:
            return JSONResponse(
                status_code=200,
                content={
                    "candidate_id": candidate_id,
                    "candidate_name": candidate_name,
                    "candidate_skills": sorted(candidate_skills),
                    "recommended_jobs": []
                }
            )

        recommended = []
        for row in job_rows:
            job_id = row[0]
            job_title = row[1]
            job_desc = row[2]
            job_skills = _normalize_skills(row[3])
            total_required = len(job_skills)

            matched = sorted(job_skills & candidate_skills)
            missing = sorted(job_skills - candidate_skills)
            match_pct = round((len(matched) / total_required * 100) if total_required > 0 else 0.0, 1)

            recommended.append({
                "rank": 0,  # filled below
                "job_id": job_id,
                "job_title": job_title,
                "job_description": job_desc or "",
                "total_required_skills": total_required,
                "matched_skills": matched,
                "missing_skills": missing,
                "matched_count": len(matched),
                "missing_count": len(missing),
                "match_percentage": match_pct
            })

        # Sort by match_percentage desc, then job_id asc as tiebreaker
        recommended.sort(key=lambda x: (-x["match_percentage"], x["job_id"]))
        for i, item in enumerate(recommended):
            item["rank"] = i + 1

        return JSONResponse(
            status_code=200,
            content={
                "candidate_id": candidate_id,
                "candidate_name": candidate_name,
                "candidate_skills": sorted(candidate_skills),
                "recommended_jobs": recommended
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        if 'conn' in locals():
            release_connection(conn)
        return JSONResponse(status_code=500, content={"error": str(e)})
