from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "data/faiss_index"
import faiss
import os

load_dotenv()
# MODEL_NAME = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")




def load_documents(data_dir: str = "data/docs"):
    """Load PDFs from data/docs and split into chunks."""
    os.makedirs(data_dir, exist_ok=True)
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)




def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    from langchain_community.vectorstores import FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)




# def load_vector_store():
#     embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
#     return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)




def load_vector_store():
    
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    # If folder missing → no data
    if not os.path.exists(INDEX_DIR):
        return create_empty_vector_store(embeddings)

    index_path = os.path.join(INDEX_DIR, "index.faiss")
    store_path = os.path.join(INDEX_DIR, "index.pkl")

    # If files missing → no data
    if not (os.path.exists(index_path) and os.path.exists(store_path)):
        return create_empty_vector_store(embeddings)

    # Load FAISS
    try:
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return create_empty_vector_store(embeddings)


def create_empty_vector_store(embeddings):
    # Determine embedding dimension
    dimension = len(embeddings.embed_query("test"))

    # Empty FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Proper docstore
    docstore = InMemoryDocstore({})

    # Return empty vector store
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id={}
    )