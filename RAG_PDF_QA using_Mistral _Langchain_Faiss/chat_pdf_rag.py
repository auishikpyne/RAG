#complete code

import gradio as gr
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from nltk.tokenize import sent_tokenize
from ctransformers import AutoModelForCausalLM

# Load PDF and process text
def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page_text := page.extract_text():
            text += page_text.replace("\n", " ").strip() + " "
    return text

# Split into chunks using NLTK
def split_by_sentences(text, sentences_per_chunk=4, overlap=1):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk - overlap):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks

# Load model (once)
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=32  # Set 0 if CPU-only
)

# Create a global FAISS DB variable
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = None

# Step 1: Load PDF, extract chunks, store in FAISS
def process_pdf(file):
    global db

    text = load_pdf_text(file.name)
    chunks = split_by_sentences(text, 3, 1)
    db = FAISS.from_texts(chunks, embedding=embedding_model)
    return "PDF processed and indexed."

# Step 2: Handle user question
def answer_question(query):
    if not db:
        return "Please upload and process a PDF first."
    docs = db.similarity_search(query, k=2)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""Answer the question very precisely based only on the context below.

Context:
{context}

Question:
{query}

Answer:"""
    response = llm(prompt)
    return response

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("### 🧠 Chat with PDF")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF")
        upload_btn = gr.Button("Process PDF")

    query_input = gr.Textbox(label="Ask a question about the PDF")
    submit_btn = gr.Button("Get Answer")
    output = gr.Textbox(label="Answer")

    upload_btn.click(process_pdf, inputs=pdf_input, outputs=output)
    submit_btn.click(answer_question, inputs=query_input, outputs=output)

demo.launch()
