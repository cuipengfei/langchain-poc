# vector_store_setup.py
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_postgres import PGVector


# 创建向量存储实例
def create_vector_store() -> VectorStore:
    #  docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
    return PGVector(
        embeddings=DashScopeEmbeddings(model="text-embedding-v3"),
        collection_name="ai_learning",
        connection="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
        use_jsonb=True
    )
