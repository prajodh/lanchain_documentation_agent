from dotenv import load_dotenv
load_dotenv()

from langchain.chains.retrieval import create_retrieval_chain

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings



INDEX_NAME = "lancghain-docs"

def run_llm(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    doc_search = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

    retrival_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_document_chain = create_stuff_documents_chain(chat, retrival_prompt)
    qa = create_retrieval_chain(
        retriever = doc_search.as_retriever(), combine_docs_chain=stuff_document_chain
    )
    result = qa.invoke({"input":query})
    return result


if __name__ == "__main__":
    res = run_llm("what is a langchain chain?")
    print(res["answer"])