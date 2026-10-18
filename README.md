# Gemini RAG Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) system built with **LangChain**, **ChromaDB**, and **Google Gemini models** (`gemini-3.5-flash` and `gemini-embedding-001`).

---

## Architecture Overview

```
[ Documents in docs/ ]
         │
         ▼
[ RecursiveCharacterTextSplitter ]
         │
         ▼
[ GoogleGenerativeAIEmbeddings ]
         │
         ▼
[ Chroma Vector Store (db/chroma_db) ]
         │
         │  (User Query)
         ▼
[ Similarity Search (Cosine) ]
         │
         ▼
[ Context Augmentation & Prompt Template ]
         │
         ▼
[ Gemini 3.5 Flash LLM ]
         │
         ▼
[ StrOutputParser ] ──► Final Synthesized Answer
```

---

## Features

- **Document Ingestion**: Automatically loads `.txt` documents from `docs/` with auto-detected encoding.
- **Smart Chunking**: Splits content using `RecursiveCharacterTextSplitter` with configurable chunk size and overlap.
- **Vector Storage**: Indexes and persists document embeddings using ChromaDB with cosine similarity search.
- **Duplicate Prevention**: Ingestion pipeline safely refreshes previous indexes to prevent duplicate embeddings.
- **Answer Generation**: Generates grounded, context-aware answers using Google Gemini (`gemini-3.5-flash`) via LangChain Expression Language (LCEL).

---

## Project Structure

```
├── docs/                     # Source documents for knowledge base
│   └── sample_doc.txt
├── ingestion_pipeline.py     # Document loader, chunker, and vector store creator
├── retrieval_pipeline.py     # Vector retriever and Gemini generation pipeline
├── .env.example              # Environment variable template
├── .gitignore                # Protects secrets (.env), venv, and local DB
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.10 or higher
- A Google Gemini API Key (Get one from [Google AI Studio](https://aistudio.google.com/app/apikey))

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

---

## Usage

### 1. Ingest Documents
Place your knowledge base `.txt` files inside the `docs/` directory, then run:
```bash
python ingestion_pipeline.py
```
This loads your documents, splits them into chunks, computes embeddings, and stores them in `db/chroma_db`.

### 2. Run Retrieval & Ask Questions
Run the RAG pipeline with a custom question:
```bash
python retrieval_pipeline.py "What are the main project goals?"
```
Or run without arguments to execute the default test query:
```bash
python retrieval_pipeline.py
```

---

## Tech Stack
- **Orchestration**: [LangChain](https://github.com/langchain-ai/langchain)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Embedding Model**: `models/gemini-embedding-001`
- **LLM / Generation**: `models/gemini-3.5-flash`

---

## License
This project is open-source and available under the [MIT License](LICENSE).
