from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from environment_loader import load_environment
from vector_store_setup import create_vector_store


# 加载并拆分文档
def load_and_split_documents(file_path: str) -> List[Document]:
    loader: TextLoader = TextLoader(file_path, encoding="utf-8")
    docs: List[Document] = loader.load()
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)


if __name__ == "__main__":
    load_environment()  # 加载环境变量

    splits: List[Document] = load_and_split_documents("introduction.txt")  # 加载并拆分文档

    vectorstore: VectorStore = create_vector_store()  # 创建向量存储
    vectorstore.add_documents(splits)  # 添加文档到向量存储

    # 执行相似性搜索并打印结果
    documents: List[Document] = vectorstore.similarity_search("豚母升木的研究内容有哪些？其引用的资料有哪些？")
    print(documents)
