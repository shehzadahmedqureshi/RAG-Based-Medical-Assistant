from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv
load_dotenv()


DATA_PATH = "data/"
VECTOR_DIR = "vectorstore/chroma_db"


def load_pdf_files(data: str):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    return documents


def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


if __name__ == "__main__":
    # Legacy script: bulk index all PDFs in DATA_PATH into Chroma.
    documents = load_pdf_files(data=DATA_PATH)
    text_chunks = create_chunks(extracted_data=documents)

    embedding_model = get_embedding_model()
    db = Chroma.from_documents(
        text_chunks,
        embedding_model,
        persist_directory=VECTOR_DIR,
        collection_name="medibot_documents",
    )
    db.persist()