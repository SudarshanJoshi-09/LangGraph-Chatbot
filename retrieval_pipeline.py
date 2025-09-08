from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"


# Load embeddings and vectore store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    embedding_function=embedding_model,
    persist_directory=persistent_directory,
    collection_name="spacex_docs",
    collection_metadata={"hnsw:space": "cosine"},
)

# Create retriever
retriever = db.as_retriever(search_kwargs={"k": 3})


# Query the vector store
def query_vector_store(query):
    """Query the vector store and return the relevant documents"""
    print(f"Querying the vector store with query: {query}")
    relevant_docs = retriever.invoke(query)
    return relevant_docs

if __name__ == "__main__":
    query = "What is the future of SpaceX?"
    relevant_docs = query_vector_store(query)
    print(relevant_docs[0].page_content)










