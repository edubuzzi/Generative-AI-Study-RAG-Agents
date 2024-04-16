from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def file_to_retriever(file_path):

    text_loader_kwargs = {'autodetect_encoding': True}
    loader = DirectoryLoader(file_path, glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()

    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

    documents = text_splitter.split_documents(docs)

    vector = Chroma.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    return retriever