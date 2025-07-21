# PDFSage: AI-Powered PDF Chat & Summarizer

## Overview

**PDFSage** is a smart, AI-powered tool that enables users to upload any PDF document and interactively ask questions about its content using a conversational interface. Built with **LangChain**, **FAISS**, **Gradio**, and **Transformers**, this system extracts and indexes the content of a PDF to support efficient and accurate retrieval-based question answering.

Designed with modularity and scalability in mind, PDFSage can also be integrated as a backend for an **MCP server** through **Gradle-based deployment**, and is live at:  
**[https://huggingface.co/spaces/Agents-MCP-Hackathon/PDFSage](https://huggingface.co/spaces/Agents-MCP-Hackathon/PDFSage)**

## Key Features

- **PDF Upload & Parsing**: Users can upload any PDF document; the system extracts its contents and preprocesses them for downstream use.
- **Semantic Text Chunking**: The document is split into meaningful overlapping chunks using `RecursiveCharacterTextSplitter`, ensuring context is preserved during retrieval.
- **FAISS Vector Store**: Efficient indexing of text chunks using embeddings (`all-MiniLM-L6-v2`) for rapid retrieval.
- **LLM-Powered QA**: Uses `google/flan-t5-base` deployed through HuggingFace Transformers for question answering.
- **Chat-Based Interface**: An intuitive Gradio interface supports natural language interaction and contextual follow-ups.
- **Stateless Design**: Each session is isolated, and PDFs are processed dynamically at runtime.

## Tech Stack

- **Python**
- **LangChain** (text splitting, chaining)
- **Gradio** (web interface)
- **FAISS** (vector store)
- **HuggingFace Transformers & Pipelines**
- **Google FLAN-T5 Base** (LLM for text generation)
- **HuggingFace Spaces** (for deployment)

## How It Works

### Step-by-Step Pipeline

1. **Upload PDF**  
   The user uploads a PDF through the Gradio interface.

2. **Text Extraction**  
   The `PdfReader` module extracts raw text from each page.

3. **Chunking & Embedding**  
   Text is split using `RecursiveCharacterTextSplitter`, then converted to vector embeddings with `all-MiniLM-L6-v2`.

4. **Vector Indexing**  
   The system creates a FAISS vector store of the chunks for efficient retrieval.

5. **LLM Integration**  
   On user query, relevant chunks are retrieved and passed into a `RetrievalQA` chain using `google/flan-t5-base`.

6. **Response Generation**  
   The model generates a natural language answer based on the contextual document chunks.

7. **Chat Display**  
   The user sees the answer within a Gradio chatbot interface, and may continue querying the same PDF.

## Deployment

PDFSage is publicly available and deployed at:  
**[https://huggingface.co/spaces/Agents-MCP-Hackathon/PDFSage](https://huggingface.co/spaces/Agents-MCP-Hackathon/PDFSage)**

This version is compatible with MCP environments and built using a Gradle project configuration for scalable cloud integration.

## Getting Started Locally

### Prerequisites

- Python 3.9+
- pip
- `torch`, `transformers`, `langchain`, `faiss-cpu`, `gradio`, `pypdf`, `sentence-transformers`

### Installation

```
git clone https://github.com/your-username/pdfsage.git
cd pdfsage
pip install -r requirements.txt
```

### Running the Application

```
python app.py
```
The application will start a local server via Gradio, accessible at `http://localhost:7860`

## Future Enhancements

* Add multilingual support for PDFs written in non-English languages.
* Introduce summarization capabilities alongside QA (e.g., section-wise summaries).
* Include PDF metadata extraction and visual table parsing.
* Store processed PDFs and chat history for persistent sessions.
* Add support for uploading multiple PDFs and context switching.
* Add better chunking techniques and increase the chunk lenght for better performace.


## Limitations

* The current version processes one PDF at a time.
* PDF files with complex formatting (e.g., tables, figures) may result in partial or incorrect text extraction.
* Chat history is non-persistent and resets with each session reload.
* The performance depends on model inference speed; may lag under heavy load without GPU support.
* Doesn't perform well on huge pdf files
