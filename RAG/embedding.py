from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

MODEL_NAME = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("data/faiss_index")


def load_vector_store():
    """
    Load an existing FAISS vector store from local storage using the same embedding model.
    """
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)