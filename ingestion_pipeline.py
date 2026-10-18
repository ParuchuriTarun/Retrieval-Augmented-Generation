import os
import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    print("Loading all text files...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist please create it and add your company files")
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}
    )
    documents = loader.load()
    if len(documents) == 0:
        raise ValueError(f"No txt documents found in {docs_path}")
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}")
        print(f" Source: {doc.metadata.get('source', 'Unknown')}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Metadata: {doc.metadata}")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content: ")
            print(chunk.page_content)
            print("-" * 50)
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db", reset_existing=True):
    print("Creating embeddings and storing in ChromaDB...")
    if reset_existing and os.path.exists(persist_directory):
        print(f"Clearing existing vector store at {persist_directory} to avoid duplicate chunks...")
        shutil.rmtree(persist_directory, ignore_errors=True)

    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    print("--- Creating Vector Store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished Creating Vector Store ---")
    print(f"Vector store is created and saved at {persist_directory}")
    return vectorstore

def main():
    print("Main Function")
    documents = load_documents(docs_path="docs")
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()