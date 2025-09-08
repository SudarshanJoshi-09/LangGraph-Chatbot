import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()




def load_documents(docs_path):
    """Load the documents from the directory"""
    print(f"Loading documents from {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Documents path {docs_path} does not exist")

    
    loader = DirectoryLoader(
        path=docs_path, 
        glob="*.txt", 
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
        )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No documents found in {docs_path}")

    return documents


def chunk_documents(documents):
    """Split the documents into smaller chunks"""
    print(f"Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
         chunk_size=1000,      
        chunk_overlap=150, 
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def create_vectore_store(chunks):
    """Create and persist ChromaDB vector store"""
    print(f"Creating embeddings and storing in chromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    persistent_directory = "db/chroma_db"

    # Create the vector store
    print(f"--- Creating Vectore store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persistent_directory,
        collection_name="spacex_docs",
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"--- Vector store created successfully ---")
    print(f"Vector store created and saved to {persistent_directory}")

    return vectorstore


def main():
    print("Starting the ingestion pipeline...")


    # 1. Load the documents
    documents = load_documents(docs_path="docs")

    # 2. chunking the files
    chunks = chunk_documents(documents)


    # 3. Embeddings and storing in Vector DB
    vectorstore = create_vectore_store(chunks)





if __name__ == "__main__":
    main()
