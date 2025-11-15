# from langchain.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def load_documents():
#     loader = DirectoryLoader("./data/docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
#     docs = loader.load()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return splitter.split_documents(docs)
