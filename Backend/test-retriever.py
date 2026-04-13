from retriever import HybridRetriever

retriever = HybridRetriever(
    faiss_index_path="FAISS_Index/candidate_index",
    snippet_metadata_path="snippet_metadata.json",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_pass="khubaib123"
)

queries = [
    "Looking for Python developer with ML and NLP experience",
    "Senior data scientist role with leadership skills"
]

for q in queries:
    results = retriever.retrieve(q)
    print(f"Query: {q}")
    for r in results:
        print(f"Candidate: {r['candidate_id']}, Score: {r['composite_score']}")
        print(f"Snippet: {r['snippet']}")
        print(f"Graph Path: {r['graph_path']}")
        print("------")