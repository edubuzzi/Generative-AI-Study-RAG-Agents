from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vectordb import url_to_retriever
from chatgpt import llm

prompt = ChatPromptTemplate.from_template("""Answer the question based only on the context:
{context}
Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = url_to_retriever("https://en.wikipedia.org/wiki/Bitcoin")

retriever_chain = create_retrieval_chain(retriever, document_chain)

response = retriever_chain.invoke({"input": "When's the next halving of Bitcoin"})
print(response["answer"])