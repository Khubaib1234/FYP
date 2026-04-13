from sentence_transformers import SentenceTransformer
import faiss
from neo4j import GraphDatabase
import numpy as np

class HybridRetriever:
    def __init__(self, faiss_index_path, snippet_metadata_path, neo4j_uri, neo4j_user, neo4j_pass):
        # Load FAISS index
        self.faiss_index = faiss.read_index(faiss_index_path)
        
        # Load snippet metadata (maps index -> candidate, text, doc_id)
        import json
        with open(snippet_metadata_path, "r") as f:
            self.snippet_metadata = json.load(f)
        
        # Load embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # or your custom model
        
        # Neo4j driver
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        
        # Scoring weights
        self.alpha = 0.6
        self.beta = 0.4

    def embed_query(self, query_text):
        return self.embed_model.encode(query_text).astype('float32')

    def search_faiss(self, query_vector, top_k=20):
        D, I = self.faiss_index.search(np.array([query_vector]), top_k)
        
        top_snippets = []
        top_sim = []
        for idx, d in zip(I[0], D[0]):
            if idx == -1:
                continue
            str_idx = str(idx)
            if str_idx in self.snippet_metadata:
                top_snippets.append(self.snippet_metadata[str_idx])
                top_sim.append(1 / (1 + d))  # normalize L2 distance
                
        return top_snippets, top_sim

    def get_graph_score(self, candidate_id, query_entities):
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Candidate {id: $cid})-[r]-(n)
                    RETURN type(r) AS rel_type, n.name AS neighbor
                """, cid=candidate_id)
                
                score = 0
                paths = []
                for record in result:
                    paths.append(f"{candidate_id} -[{record['rel_type']}]-> {record['neighbor']}")
                    # Simple scoring: count matches with query entities
                    if record['neighbor'].lower() in query_entities:
                        score += 1
                return score, paths
        except Exception as e:
            print(f"Graph query failed for {candidate_id}: {e}")
            return 0, []

    def retrieve(self, query_text, top_k=5):
        # 1. Extract query entities (skills/roles/traits) if needed
        query_entities = [word.lower() for word in query_text.split()]
        query_vector = self.embed_query(query_text)
        
        # 2. FAISS initial fetch (get more than needed for graph reranking, max index bound safety)
        fetch_k = max(20, top_k * 2)
        total_in_index = self.faiss_index.ntotal
        fetch_k = min(fetch_k, total_in_index) if total_in_index > 0 else fetch_k
        top_snippets, dense_sim = self.search_faiss(query_vector, top_k=fetch_k)
        
        # 3. Graph expansion and de-duplication
        results = []
        seen_candidates = set()
        
        for snippet, sim in zip(top_snippets, dense_sim):
            candidate_id = snippet["candidate_id"]
            if candidate_id in seen_candidates:
                continue
            
            graph_score, graph_paths = self.get_graph_score(candidate_id, query_entities)
            composite = self.alpha * sim + self.beta * graph_score
            results.append({
                "candidate_id": candidate_id,
                "snippet": snippet["text"][:200],  # 2-sentence summary
                "graph_path": graph_paths[0] if graph_paths else None,
                "dense_sim": sim,
                "graph_score": graph_score,
                "composite_score": composite
            })
            seen_candidates.add(candidate_id)
        
        # 4. Rank top 5
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return results[:top_k]