# vector_store_setup.py
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.vectorstores import VectorStore


def create_vector_store() -> VectorStore:
    return Chroma(
        collection_name="ai_learning",
        embedding_function=DashScopeEmbeddings(model="text-embedding-v3"),
        persist_directory="vectordb"
    )
