from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loader = TextLoader("introduction.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma(
    collection_name="ai_learning",
    embedding_function=DashScopeEmbeddings(model="text-embedding-v3"),
    persist_directory="vectordb"
)
vectorstore.add_documents(splits)

documents = vectorstore.similarity_search("豚母升木的研究内容有哪些？其引用的都资料有哪些？")
print(documents)
