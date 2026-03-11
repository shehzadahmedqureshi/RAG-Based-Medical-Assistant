import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


def get_vectorstore() -> Chroma:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(
        collection_name="medibot_documents",
        embedding_function=embedding_model,
        persist_directory="vectorstore/chroma_db",
    )
    return db


def build_rag_chain():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")

    groq_model_name = "llama-3.1-8b-instant"  # Change to any supported Groq model

    llm = ChatGroq(
        model=groq_model_name,
        temperature=0.5,
        max_tokens=512,
        api_key=groq_api_key,
    )

    db = get_vectorstore()

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    rag_chain = create_retrieval_chain(
        db.as_retriever(search_kwargs={"k": 3}), combine_docs_chain
    )
    return rag_chain


if __name__ == "__main__":
    rag_chain = build_rag_chain()
    user_query = input("Write Query Here: ")
    response = rag_chain.invoke({"input": user_query})
    print("RESULT: ", response["answer"])
    print("\nSOURCE DOCUMENTS:")
    for doc in response["context"]:
        print(f"- {doc.metadata} -> {doc.page_content[:200]}...")
