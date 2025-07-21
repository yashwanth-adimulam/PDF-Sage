from pypdf import PdfReader
import gradio as gr
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text content from a given PDF file.
    ... (rest of the function, no changes here yet)
    """
    text_content = ""
    if pdf_path is None:
        return "Please upload a PDF file."
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text_content += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return f"Error: Could not read PDF. {e}"
    return text_content


def process_pdf_to_chunks(pdf_path: str) -> list[str]:
    """
    Extracts text from a PDF and splits it into manageable chunks.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        list[str]: A list of text chunks.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text or "Error:" in raw_text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def create_vector_store(chunks: list[str]):
    """
    Creates a FAISS vector store from a list of text chunks.

    Args:
        chunks (list[str]): A list of text chunks.

    Returns:
        FAISS: An initialized FAISS vector store.
    """
    if not chunks:
        return None

    embeddings = get_embedding_model()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


def setup_pdf_for_rag(pdf_file_obj):
    """
    Processes an uploaded PDF, creates a vector store, and makes it globally available.
    This function is designed to be called when a new PDF is uploaded.
    """
    global pdf_vector_store, current_pdf_path

    if pdf_file_obj is None:
        pdf_vector_store = None
        current_pdf_path = None
        return "Please upload a PDF file to process.", "No PDF processed."

    new_pdf_path = pdf_file_obj.name

    if new_pdf_path != current_pdf_path or pdf_vector_store is None:
        print(f"Processing new PDF: {new_pdf_path}")
        chunks = process_pdf_to_chunks(new_pdf_path)
        if not chunks:
            pdf_vector_store = None
            current_pdf_path = None
            return f"Error processing PDF or no text found in {os.path.basename(new_pdf_path)}.", ""

        pdf_vector_store = create_vector_store(chunks)
        current_pdf_path = new_pdf_path

        status_message = "PDF processed and vector store created successfully!"
        return status_message, f"Ready to answer questions about {os.path.basename(new_pdf_path)}."
    else:
        status_message = f"PDF {os.path.basename(current_pdf_path)} already processed."
        return status_message, f"Ready to answer questions about {os.path.basename(current_pdf_path)}."


def ask_pdf_rag(query: str) -> str:
    """
    Answers a question using the loaded PDF's vector store and the LLM.
    """
    global pdf_vector_store

    if pdf_vector_store is None:
        return "Please upload and process a PDF first."
    if not query.strip():
        return "Please enter a question."

    llm = get_llm_model()

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context:
    {context}

    Question: {question}
    Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=pdf_vector_store.as_retriever(),
        return_source_documents=False,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    try:
        result = qa_chain.invoke({"query": query})
        return result.get("result", "Could not get an answer from the PDF.")
    except Exception as e:
        print(f"Error during RAG query: {e}")
        return f"An error occurred while trying to answer: {e}"


pdf_vector_store = None
current_pdf_path = None

embedding_model = None
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model (all-MiniLM-L6-v2)... This may take a moment.")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Embedding model loaded.")
    return embedding_model


llm_model = None
def get_llm_model():
    global llm_model
    if llm_model is None:
        print("Loading LLM (Google/flan-t5-base)... This will take a moment.")
        model_id = "google/flan-t5-base"
        pipe = pipeline(
            "text2text-generation",
            model=model_id,
            max_new_tokens=500,
            device=-1,
            model_kwargs={"max_length": 512}
        )
        llm_model = HuggingFacePipeline(pipeline=pipe)
        print("LLM loaded.")
    return llm_model


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # PDFSage: Smart PDF Chat & Summarizer
        Upload a PDF, let the AI process it, and then ask questions or request summaries!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_file = gr.File(label="1. Upload your PDF")
            process_pdf_button = gr.Button("Process PDF")
            pdf_status_output = gr.Textbox(label="PDF Processing Status", interactive=False, lines=2)

        with gr.Column(scale=2):
            chat_history = gr.Chatbot(height=350, type="messages")
            query_input = gr.Textbox(label="Ask a question or request a summary about the PDF...")
            submit_query_button = gr.Button("Ask PDF")
            clear_button = gr.Button("Clear Chat")

    process_pdf_button.click(
        fn=setup_pdf_for_rag,
        inputs=[pdf_file],
        outputs=[pdf_status_output, gr.State(None)]
    )

    def respond_to_query(message, chat_history_list):
        if pdf_vector_store is None:
            chat_history_list.append({"role": "user", "content": message})
            chat_history_list.append({"role": "assistant", "content": "Please upload and process a PDF first."})
            return "", chat_history_list

        chat_history_list.append({"role": "user", "content": message})

        response = ask_pdf_rag(message)
        chat_history_list.append({"role": "assistant", "content": response})
        return "", chat_history_list

    query_input.submit(
        fn=respond_to_query,
        inputs=[query_input, chat_history],
        outputs=[query_input, chat_history]
    )
    submit_query_button.click(
        fn=respond_to_query,
        inputs=[query_input, chat_history],
        outputs=[query_input, chat_history]
    )

    clear_button.click(lambda: None, None, chat_history, queue=False)
    clear_button.click(lambda: "", None, query_input, queue=False)

if __name__ == "__main__":
    get_embedding_model()
    get_llm_model()
    demo.launch()