import streamlit as st
import requests
import time
import math
import json
import networkx as nx
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

# -------------------------
# Configuration
# -------------------------
st.set_page_config(
    page_title="TalentScope AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

UPLOAD_API_URL         = "http://127.0.0.1:8000/upload-resume/"
GRAPH_API_URL          = "http://127.0.0.1:8000/graph-data"
CANDIDATES_API_URL     = "http://127.0.0.1:8000/candidates"
SEARCH_API_URL         = "http://127.0.0.1:8000/search"
JOBS_API_URL           = "http://127.0.0.1:8000/jobs"
SKILL_GAP_API_URL      = "http://127.0.0.1:8000/skill-gap"
RANK_CANDIDATES_API_URL = "http://127.0.0.1:8000/rank-candidates"
RECOMMEND_JOBS_API_URL = "http://127.0.0.1:8000/recommend-jobs"

# -------------------------
# Page Header
# -------------------------
st.title("TalentScope AI - Resume Analyzer")
st.markdown("---")

# -------------------------
# Pipeline Phases
# -------------------------
pipeline_phases = [
    "Upload",
    "Cleaning",
    "Structured JSON Extraction",
    "Trait Inference",
    "FAISS Update",
    "Database + Graph Sync"
]

# -------------------------
# Cached data fetchers
# (defined early so we can call .clear() after upload)
# -------------------------
@st.cache_data(ttl=300)
def fetch_all_candidates():
    try:
        resp = requests.get(CANDIDATES_API_URL)
        if resp.status_code == 200:
            return resp.json()
        return []
    except:
        return []

@st.cache_data(ttl=300)
def fetch_graph_data():
    try:
        resp = requests.get(GRAPH_API_URL)
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None

@st.cache_data(ttl=60)
def fetch_jobs():
    try:
        resp = requests.get(JOBS_API_URL)
        if resp.status_code == 200:
            return resp.json()
        return []
    except:
        return []

# -------------------------
# Section 1: Resume Upload
# -------------------------
st.header("1️⃣ Upload Resume & Pipeline Status")
st.info("Upload a candidate resume and watch the pipeline progress.")

uploaded_file = st.file_uploader(
    "Drag & drop a resume here or select a file",
    type=["pdf", "docx", "txt"]
)

# --- Duplicate Upload Fix ---
# Guard the upload with session-state flags so that widget interactions
# (typing in search, moving slider, clicking any button) that cause Streamlit
# to re-run the script do NOT trigger a second /upload-resume/ call.
if "upload_processed" not in st.session_state:
    st.session_state.upload_processed = False
    st.session_state.last_uploaded_filename = None
    st.session_state.upload_response = None

# Genuinely new file → reset state
new_file_detected = (
    uploaded_file is not None
    and uploaded_file.name != st.session_state.last_uploaded_filename
)
if new_file_detected:
    st.session_state.upload_processed = False
    st.session_state.last_uploaded_filename = uploaded_file.name
    st.session_state.upload_response = None

if uploaded_file and not st.session_state.upload_processed:
    progress_bars = {phase: st.progress(0, text=f"{phase} - Pending") for phase in pipeline_phases}

    try:
        progress_bars["Upload"].progress(50, text="Upload - In progress")
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = requests.post(UPLOAD_API_URL, files=files)
        progress_bars["Upload"].progress(100, text="Upload - Completed")

        if response.status_code == 200:
            data = response.json()
            st.session_state.upload_response = data
            st.session_state.upload_processed = True

            for phase in pipeline_phases[1:]:
                progress_bars[phase].progress(50, text=f"{phase} - In progress")
                time.sleep(0.5)
                progress_bars[phase].progress(100, text=f"{phase} - Completed")

            # ── Cache invalidation ─────────────────────────────────────────
            # Clear cached candidates and graph so the new upload is visible
            # immediately without waiting for TTL to expire.
            fetch_all_candidates.clear()
            fetch_graph_data.clear()

            st.success("✅ Resume processed successfully!")
            with st.expander("View Structured JSON"):
                st.json(data["structured_output"])
            with st.expander("View Trait Scores"):
                st.json(data["traits_output"])
        else:
            st.error(f"❌ Upload failed: {response.text}")

    except Exception as e:
        st.error(f"❌ Error during upload: {e}")

elif uploaded_file and st.session_state.upload_processed and st.session_state.upload_response:
    st.success("✅ Resume already processed.")
    with st.expander("View Structured JSON"):
        st.json(st.session_state.upload_response["structured_output"])
    with st.expander("View Trait Scores"):
        st.json(st.session_state.upload_response["traits_output"])

st.markdown("---")

# -------------------------
# Section 2: Candidate Search
# -------------------------
st.header("🔍 Candidate Search")
st.info("Query the hybrid retriever to find the best candidate matches based on skills and experience.")

search_query = st.text_input("Enter search query (e.g., 'Machine Learning Engineer with Python')")
top_k = st.slider("Number of top candidates to retrieve", min_value=1, max_value=50, value=3)

if st.button("Search Candidates"):
    if search_query.strip():
        with st.spinner("Searching..."):
            try:
                search_payload = {"query": search_query, "top_k": top_k}
                response = requests.post(SEARCH_API_URL, json=search_payload)
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    if results:
                        st.success(f"Found {len(results)} matching candidates!")
                        for i, res in enumerate(results):
                            with st.expander(f"Candidate Match {i+1} (Score: {res['composite_score']:.2f})"):
                                st.write(f"**Candidate ID:** {res['candidate_id']}")
                                st.write(f"**Snippet:** {res['snippet']}...")
                                if res.get("graph_path"):
                                    st.write(f"**Graph Connection:** {res['graph_path']}")
                                st.write(f"*(Vector Score: {res['dense_sim']:.2f}, Graph Score: {res['graph_score']})*")
                    else:
                        st.info("No matches found for your query.")
                else:
                    st.error(f"Search failed: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
    else:
        st.warning("Please enter a search query.")

st.markdown("---")

# -------------------------
# Section 3: Candidate Graph
# -------------------------
st.header("2️⃣ Candidate Graph")
st.info("Interactive candidate graph (click a candidate node to view details). Press 'Refresh Graph' after upload if new nodes don't appear.")

# Manual refresh button clears the graph cache immediately
if st.button("🔄 Refresh Graph"):
    fetch_graph_data.clear()
    st.rerun()

graph_data = fetch_graph_data()
selected_candidate_id = None

if graph_data and graph_data.get("nodes"):
    G = nx.Graph()
    node_map = {}
    for node in graph_data["nodes"]:
        node_id = node["id"]
        G.add_node(node_id, **node)
        node_map[node_id] = node

    for edge in graph_data["edges"]:
        G.add_edge(edge["source"], edge["target"], label=edge["label"])

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text, node_color, node_ids = [], [], [], [], []
    for n, data in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(data.get('label', ''))
        # ── Fix: Neo4j returns 'type' as the first label string (not a property)
        node_type = data.get('type', '')
        node_color.append('#1DB954' if node_type == 'Candidate' else '#FFD700')
        node_ids.append(n)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(color=node_color, size=20, line_width=2),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111"
        )
    )

    clicked_nodes = plotly_events(fig, click_event=True)
    if clicked_nodes:
        point_idx = clicked_nodes[0]['pointIndex']
        selected_candidate_id = node_ids[point_idx]
else:
    st.warning("No graph data available. Upload resumes to build the graph.")

st.markdown("---")

# -------------------------
# Section 4: Candidate Details
# -------------------------
st.header("3️⃣ Candidate Details")
st.info("Cards with structured JSON and trait scores. Press 'Refresh Candidates' if a newly uploaded resume isn't visible.")

if st.button("🔄 Refresh Candidates"):
    fetch_all_candidates.clear()
    st.rerun()

candidate_data_list = fetch_all_candidates()

# Deduplicate (keep by candidate_id)
unique_candidates = {}
for c in candidate_data_list:
    unique_candidates[str(c['candidate_id'])] = c
candidate_data_list = list(unique_candidates.values())

# Filter by clicked graph node
if selected_candidate_id:
    candidate_data_list = [c for c in candidate_data_list if str(c['candidate_id']) == str(selected_candidate_id)]

cols_per_row = 3
num_rows = math.ceil(len(candidate_data_list) / cols_per_row)

for row in range(num_rows):
    cols = st.columns(cols_per_row, gap="medium")
    for i in range(cols_per_row):
        idx = row * cols_per_row + i
        if idx >= len(candidate_data_list):
            break
        data = candidate_data_list[idx]
        candidate_id = data['candidate_id']
        with cols[i]:
            st.markdown(f"""
            <div style="background-color:#222222;padding:15px;border-radius:10px;box-shadow:2px 2px 10px rgba(0,0,0,0.5);">
            <h3 style="color:#1DB954;">{data['structured_json'].get('name','Candidate')}</h3>
            <p style="color:#E0E0E0;">Email: {data['structured_json'].get('email','N/A')}</p>
            <p style="color:#E0E0E0;">Phone: {data['structured_json'].get('phone','N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

            traits = data.get("traits", {})
            if traits:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=list(traits.values()),
                    theta=list(traits.keys()),
                    fill='toself',
                    marker_color='#1DB954'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    paper_bgcolor='#222222',
                    font_color='#E0E0E0',
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True, key=f"radar_{candidate_id}")

            with st.expander("Structured JSON"):
                st.json(data["structured_json"])

            st.download_button(
                label="Download Resume JSON",
                data=json.dumps(data["structured_json"], indent=2),
                file_name=f"{candidate_id}_structured.json",
                mime="application/json"
            )
            st.download_button(
                label="Download Traits JSON",
                data=json.dumps(data["traits"], indent=2),
                file_name=f"{candidate_id}_traits.json",
                mime="application/json"
            )

st.markdown("---")

# =============================================================================
# Section 5: Skill Gap Analysis
# =============================================================================
st.header("4️⃣ Skill Gap Analysis")
st.info("Compare job requirements against candidate skills and rank candidates by fit.")

# ── Add New Job Form ──────────────────────────────────────────────────────────
with st.expander("➕ Add a New Job Posting"):
    with st.form("add_job_form"):
        job_title_input  = st.text_input("Job Title", placeholder="e.g. Data Engineer")
        job_desc_input   = st.text_area("Job Description (optional)", placeholder="Brief description of the role...")
        job_skills_input = st.text_input(
            "Required Skills (comma-separated)",
            placeholder="e.g. Python, SQL, Docker, Spark"
        )
        submitted = st.form_submit_button("Add Job")

        if submitted:
            if not job_title_input.strip():
                st.warning("Job title is required.")
            elif not job_skills_input.strip():
                st.warning("At least one skill is required.")
            else:
                skills_list = [s.strip() for s in job_skills_input.split(",") if s.strip()]
                payload = {
                    "title": job_title_input.strip(),
                    "description": job_desc_input.strip(),
                    "skills": skills_list
                }
                try:
                    r = requests.post(JOBS_API_URL, json=payload)
                    if r.status_code == 201:
                        st.success(f"✅ Job '{job_title_input}' added (ID: {r.json()['job_id']})")
                        fetch_jobs.clear()
                    else:
                        st.error(f"Failed to add job: {r.text}")
                except Exception as ex:
                    st.error(f"Error: {ex}")

st.markdown("")

# ── Job Selector ──────────────────────────────────────────────────────────────
jobs = fetch_jobs()

if not jobs:
    st.warning("No jobs found. Add a job posting above to get started.")
else:
    job_options = {f"[{j['job_id']}] {j['title']}": j for j in jobs}
    selected_job_label = st.selectbox("🎯 Select a Job", list(job_options.keys()), key="sg_job_select")
    selected_job = job_options[selected_job_label]

    st.markdown(
        f"**Required Skills ({len(selected_job['skills'])}):** "
        + " · ".join(f"`{s}`" for s in selected_job["skills"])
    )

    st.markdown("")

    # ── Sub-section A: Individual Skill Gap ─────────────────────────────────
    st.subheader("🔎 Individual Skill Gap")

    all_candidates_raw = fetch_all_candidates()
    unique_cands = {str(c['candidate_id']): c for c in all_candidates_raw}
    all_candidates_list = list(unique_cands.values())

    if not all_candidates_list:
        st.info("No candidates found. Upload resumes first.")
    else:
        cand_options = {
            f"[{c['candidate_id']}] {c['structured_json'].get('name', 'Unknown')}": c['candidate_id']
            for c in all_candidates_list
        }
        selected_cand_label = st.selectbox("👤 Select a Candidate", list(cand_options.keys()), key="sg_cand_select")
        selected_cand_id = cand_options[selected_cand_label]

        if st.button("Analyse Skill Gap"):
            with st.spinner("Analysing..."):
                try:
                    resp = requests.get(f"{SKILL_GAP_API_URL}/{selected_job['job_id']}/{selected_cand_id}")
                    if resp.status_code == 200:
                        gap = resp.json()
                        match_pct = gap["match_percentage"]
                        gauge_color = "#1DB954" if match_pct >= 70 else "#FFA500" if match_pct >= 40 else "#E74C3C"

                        st.markdown(
                            f"""
                            <div style="background:#1a1a2e;border-radius:12px;padding:20px;margin-bottom:16px;text-align:center;">
                                <h2 style="color:{gauge_color};margin:0;">{match_pct}%</h2>
                                <p style="color:#aaa;margin:4px 0 0 0;">Skill Match for
                                <strong style="color:#fff">{gap['candidate_name']}</strong>
                                → <strong style="color:#fff">{gap['job_title']}</strong></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        col_match, col_miss = st.columns(2)
                        with col_match:
                            st.markdown(
                                f"<h4 style='color:#1DB954;'>✅ Matched Skills ({gap['matched_count']})</h4>",
                                unsafe_allow_html=True
                            )
                            for skill in gap["matched_skills"]:
                                st.markdown(
                                    f"<span style='background:#1DB95422;color:#1DB954;padding:4px 10px;"
                                    f"border-radius:20px;margin:3px;display:inline-block;'>{skill}</span>",
                                    unsafe_allow_html=True
                                )
                            if not gap["matched_skills"]:
                                st.write("_No matched skills_")

                        with col_miss:
                            st.markdown(
                                f"<h4 style='color:#E74C3C;'>❌ Missing Skills ({gap['missing_count']})</h4>",
                                unsafe_allow_html=True
                            )
                            for skill in gap["missing_skills"]:
                                st.markdown(
                                    f"<span style='background:#E74C3C22;color:#E74C3C;padding:4px 10px;"
                                    f"border-radius:20px;margin:3px;display:inline-block;'>{skill}</span>",
                                    unsafe_allow_html=True
                                )
                            if not gap["missing_skills"]:
                                st.write("_No missing skills — perfect match! 🎉_")
                    else:
                        st.error(f"Error: {resp.json().get('detail', resp.text)}")
                except Exception as ex:
                    st.error(f"Error: {ex}")

    st.markdown("")

    # ── Sub-section B: Rank Candidates ──────────────────────────────────────
    st.subheader("🏆 Rank Candidates by Skill Match")

    if st.button("Rank All Candidates"):
        with st.spinner("Ranking candidates..."):
            try:
                resp = requests.get(f"{RANK_CANDIDATES_API_URL}/{selected_job['job_id']}")
                if resp.status_code == 200:
                    ranking_data = resp.json()
                    candidates_ranked = ranking_data["candidates"]
                    if not candidates_ranked:
                        st.info("No candidates to rank.")
                    else:
                        st.success(
                            f"Ranked **{len(candidates_ranked)}** candidates for "
                            f"**{ranking_data['job_title']}** "
                            f"({ranking_data['total_required_skills']} required skills)"
                        )
                        for cand in candidates_ranked:
                            match_pct = cand["match_percentage"]
                            bar_color = "#1DB954" if match_pct >= 70 else "#FFA500" if match_pct >= 40 else "#E74C3C"
                            medal = "🥇" if cand["rank"] == 1 else "🥈" if cand["rank"] == 2 else "🥉" if cand["rank"] == 3 else f"#{cand['rank']}"
                            missing_preview = ""
                            if cand["missing_skills"]:
                                preview = ", ".join(cand["missing_skills"][:5])
                                if len(cand["missing_skills"]) > 5:
                                    preview += "…"
                                missing_preview = f"&nbsp;|&nbsp; Missing: <em>{preview}</em>"

                            st.markdown(
                                f"""
                                <div style="background:#1a1a2e;border-radius:10px;padding:14px 18px;
                                            margin-bottom:10px;border-left:5px solid {bar_color};">
                                    <div style="display:flex;justify-content:space-between;align-items:center;">
                                        <span style="font-size:1.1em;color:#fff;">
                                            {medal} &nbsp;<strong>{cand['candidate_name']}</strong>
                                            <span style="color:#888;font-size:0.85em;"> · ID {cand['candidate_id']}</span>
                                        </span>
                                        <span style="font-size:1.4em;font-weight:bold;color:{bar_color};">{match_pct}%</span>
                                    </div>
                                    <div style="background:#333;border-radius:6px;height:8px;margin-top:8px;">
                                        <div style="background:{bar_color};width:{match_pct}%;height:8px;border-radius:6px;"></div>
                                    </div>
                                    <div style="margin-top:8px;font-size:0.85em;color:#aaa;">
                                        ✅ {cand['matched_count']} matched &nbsp;|&nbsp;
                                        ❌ {cand['missing_count']} missing{missing_preview}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except Exception as ex:
                st.error(f"Error: {ex}")

st.markdown("---")

# =============================================================================
# Section 6: Job Recommendations for a Candidate
# =============================================================================
st.header("5️⃣ Job Recommendations for a Candidate")
st.info("Select a candidate to see which jobs best match their skills, ranked by compatibility.")

all_candidates_for_rec = fetch_all_candidates()
unique_cands_rec = {str(c['candidate_id']): c for c in all_candidates_for_rec}
candidates_for_rec = list(unique_cands_rec.values())

if not candidates_for_rec:
    st.warning("No candidates found. Upload resumes first.")
else:
    rec_cand_options = {
        f"[{c['candidate_id']}] {c['structured_json'].get('name', 'Unknown')}": c['candidate_id']
        for c in candidates_for_rec
    }
    selected_rec_cand_label = st.selectbox(
        "👤 Select a Candidate",
        list(rec_cand_options.keys()),
        key="rec_cand_select"
    )
    selected_rec_cand_id = rec_cand_options[selected_rec_cand_label]

    if st.button("Get Job Recommendations"):
        with st.spinner("Finding best-matching jobs..."):
            try:
                resp = requests.get(f"{RECOMMEND_JOBS_API_URL}/{selected_rec_cand_id}")
                if resp.status_code == 200:
                    rec_data = resp.json()
                    candidate_name = rec_data["candidate_name"]
                    candidate_skills = rec_data["candidate_skills"]
                    recommended_jobs = rec_data["recommended_jobs"]

                    # Show candidate skill summary
                    st.markdown(
                        f"**{candidate_name}'s Skills ({len(candidate_skills)}):** "
                        + " · ".join(f"`{s}`" for s in candidate_skills)
                    )
                    st.markdown("")

                    if not recommended_jobs:
                        st.info("No jobs available to match against. Add job postings first.")
                    else:
                        st.success(f"Found **{len(recommended_jobs)}** jobs ranked by compatibility for **{candidate_name}**")

                        for job in recommended_jobs:
                            match_pct = job["match_percentage"]
                            bar_color = "#1DB954" if match_pct >= 70 else "#FFA500" if match_pct >= 40 else "#E74C3C"
                            medal = "🥇" if job["rank"] == 1 else "🥈" if job["rank"] == 2 else "🥉" if job["rank"] == 3 else f"#{job['rank']}"

                            missing_preview = ""
                            if job["missing_skills"]:
                                preview = ", ".join(job["missing_skills"][:5])
                                if len(job["missing_skills"]) > 5:
                                    preview += "…"
                                missing_preview = f"&nbsp;|&nbsp; Skills to acquire: <em style='color:#FFA500'>{preview}</em>"

                            matched_preview = ""
                            if job["matched_skills"]:
                                matched_preview = ", ".join(job["matched_skills"][:5])
                                if len(job["matched_skills"]) > 5:
                                    matched_preview += "…"

                            st.markdown(
                                f"""
                                <div style="background:#1a1a2e;border-radius:10px;padding:16px 20px;
                                            margin-bottom:12px;border-left:5px solid {bar_color};">
                                    <div style="display:flex;justify-content:space-between;align-items:center;">
                                        <span style="font-size:1.15em;color:#fff;">
                                            {medal} &nbsp;<strong>{job['job_title']}</strong>
                                            <span style="color:#888;font-size:0.82em;"> · Job ID {job['job_id']}</span>
                                        </span>
                                        <span style="font-size:1.5em;font-weight:bold;color:{bar_color};">{match_pct}%</span>
                                    </div>
                                    <div style="background:#333;border-radius:6px;height:8px;margin-top:10px;">
                                        <div style="background:{bar_color};width:{match_pct}%;height:8px;border-radius:6px;"></div>
                                    </div>
                                    <div style="margin-top:10px;font-size:0.85em;color:#aaa;">
                                        ✅ {job['matched_count']}/{job['total_required_skills']} skills matched
                                        {"&nbsp;·&nbsp; <em style='color:#1DB954'>" + matched_preview + "</em>" if matched_preview else ""}
                                        {missing_preview}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.error(f"Error: {resp.json().get('detail', resp.text)}")
            except Exception as ex:
                st.error(f"Error: {ex}")
