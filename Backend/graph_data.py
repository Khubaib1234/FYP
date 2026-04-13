from fastapi import APIRouter
from fastapi.responses import JSONResponse
from neo4j import GraphDatabase

router = APIRouter()

# Neo4j connection
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "khubaib123"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def fetch_graph(tx):
    nodes_query = """
    MATCH (n)
    RETURN id(n) AS id, labels(n) AS labels, n.name AS name, n.type AS type
    """
    nodes_result = tx.run(nodes_query)
    nodes = []
    for record in nodes_result:
        typ = record["labels"][0]
        if typ == "Candidate":
            label = str(record["id"])        # candidate node → show ID
        else:
            label = record["name"] or typ   # skill/company → show name or type
        nodes.append({
            "id": str(record["id"]),
            "label": label,
            "type": typ
        })

    # Edges
    rels_query = "MATCH (a)-[r]->(b) RETURN id(a) AS source, id(b) AS target, type(r) AS relation"
    rels_result = tx.run(rels_query)
    edges = [
        {
            "source": str(record["source"]),
            "target": str(record["target"]),
            "label": record["relation"]
        }
        for record in rels_result
    ]

    return {"nodes": nodes, "edges": edges}

@router.get("/graph-data")
async def get_graph_data():
    try:
        with driver.session() as session:
            graph_data = session.execute_read(fetch_graph)
        return JSONResponse(status_code=200, content=graph_data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.on_event("shutdown")
def shutdown_event():
    driver.close()
