
from neo4j import GraphDatabase
import os

class Neo4jClient:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "khubaib123")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def run(self, cypher, params=None):
        with self.driver.session() as session:
            return session.run(cypher, params or {})

    def close(self):
        self.driver.close()


# create a global instance
neo4j_client = Neo4jClient()
