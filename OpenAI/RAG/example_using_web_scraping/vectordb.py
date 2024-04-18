from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# If you prefer, you can use FAISS instead of Chroma => from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

def url_to_retriever(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()

    documents = text_splitter.split_documents(docs)

    vector = Chroma.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    return retriever