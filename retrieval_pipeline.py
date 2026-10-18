import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

persist_directory = "db/chroma_db"
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)
def run_rag_pipeline(query: str, k: int = 5):
    print("=" * 60)
    print(f"RAG Pipeline - Query: {query}")
    print("=" * 60)
    relevant_docs = db.similarity_search(query, k=k)
    print(f"\n--- Retrieved Documents (Top {len(relevant_docs)}) ---")
    if not relevant_docs:
        print("No documents found. Check your ingestion pipeline and database path.")
        return None
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        print(f"\n[Document {i}] (Source: {source}):\n{doc.page_content}")
        print("-" * 40)
    context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
    prompt_template = ChatPromptTemplate.from_template(
        "You are an AI assistant answering questions based on the retrieved context below.\n"
        "Provide a concise and direct answer using only the information in the context.\n"
        "If the answer cannot be found in the context, say that the information is not available.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    print("\n--- Generating Response via Gemini LLM ---")
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
    chain = prompt_template | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    print(f"\nFinal Answer:\n{answer}\n")
    return answer

if __name__ == "__main__":
    doc_count = db._collection.count()
    print("Documents stored in Chroma:", doc_count)
    if doc_count == 0:
        print("Warning: Vector store is empty! Please run 'python ingestion_pipeline.py' first.")
    user_query = sys.argv[1] if len(sys.argv) > 1 else "What is your main goal regarding operating system development and licensing?"
    run_rag_pipeline(user_query)