Deployed here: https://customrag-agent.streamlit.app/
# Custom RAG Agent

A document-based **Retrieval-Augmented Generation (RAG)** application built with **Streamlit, LangChain, OpenAI, and FAISS**.

The application allows users to upload documents, process their contents into vector embeddings, and build a searchable knowledge base that can be used to answer questions based on the uploaded documents.

## Features

* Upload PDF documents
* Configure the OpenAI API key directly from the sidebar
* Choose between different OpenAI chat models
* Customize document chunk size
* Customize chunk overlap
* Configure the number of relevant document chunks retrieved
* Automatically split documents into smaller chunks
* Generate vector embeddings using OpenAI
* Store document embeddings using FAISS
* Efficient semantic document retrieval
* Session-based vector store management
* Rebuild the vector database when document settings change

## Tech Stack

* **Python**
* **Streamlit** — User interface
* **LangChain** — RAG pipeline and document processing
* **OpenAI** — Language models and embeddings
* **FAISS** — Vector similarity search
* **PyPDFLoader** — PDF document loading

## Project Workflow

The application follows a typical Retrieval-Augmented Generation pipeline:

```text
Upload Document
       ↓
Load PDF
       ↓
Split into Chunks
       ↓
Generate OpenAI Embeddings
       ↓
Store Vectors in FAISS
       ↓
Retrieve Relevant Chunks
       ↓
Generate Context-Aware Response
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/custom-rag-agent.git
cd custom-rag-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu pypdf
```

## Environment Variables

You can provide your OpenAI API key using an environment variable.

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Alternatively, you can enter the API key directly in the Streamlit application's sidebar.

## Running the Application

Run the application using:

```bash
streamlit run app.py
```

Then open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## Application Settings

The sidebar provides several configuration options.

### OpenAI API Key

Enter your OpenAI API key to enable embeddings and language model functionality.

### Model

Choose the OpenAI language model.

Available options:

* `gpt-3.5-turbo`
* `gpt-4`

### Chunk Size

Controls the maximum size of each document chunk.

Smaller chunks provide more precise retrieval, while larger chunks provide more context.

### Chunk Overlap

Controls how much text is shared between consecutive chunks.

Overlap helps preserve context when important information appears near the boundaries of chunks.

### Top K

Controls the number of most relevant document chunks retrieved during similarity search.

## How It Works

### Document Loading

Uploaded PDF files are temporarily saved and loaded using LangChain's `PyPDFLoader`.

```python
docs = PyPDFLoader(tmp_path).load()
```

### Text Splitting

The document is divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter
```

The chunk size and overlap can be customized through the Streamlit interface.

### Embeddings

Each document chunk is converted into a vector representation using OpenAI embeddings:

```python
OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

### Vector Database

The generated embeddings are stored inside a FAISS vector store:

```python
FAISS.from_documents(chunks, embeddings)
```

FAISS enables efficient similarity searches to find the most relevant information from uploaded documents.

## Requirements

Example `requirements.txt`:

```text
streamlit
langchain
langchain-community
langchain-core
langchain-openai
langchain-text-splitters
faiss-cpu
pypdf
openai
```

## Project Structure

```text
custom-rag-agent/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

## Future Improvements

Possible improvements for the project include:

* Support for multiple documents simultaneously
* Support for TXT, DOCX, and CSV files
* Chat interface with conversation history
* Source citations for generated answers
* Persistent vector databases
* Document preview
* Multiple embedding model options
* Streaming responses
* Chat memory
* Hybrid search
* Reranking retrieved documents
* Support for local LLMs

## Security Note

Never commit your OpenAI API key to GitHub.

Add your environment file to `.gitignore`:

```text
.env
```

## License

This project is available for educational and personal use.

---

Built with **Streamlit + LangChain + OpenAI + FAISS**.

