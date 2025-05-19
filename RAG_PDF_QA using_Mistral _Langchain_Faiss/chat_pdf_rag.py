#complete code
import gradio as gr
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from nltk.tokenize import sent_tokenize
from ctransformers import AutoModelForCausalLM
import nltk
from langchain.docstore.document import Document

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

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
    gpu_layers=0  # Set 0 if CPU-only
)

# Create a global FAISS DB variable
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = None
pdf_path = None  # Add this globally

# Step 1: Load PDF, extract chunks, store in FAISS
def process_pdf(file):
    global db, pdf_path
    pdf_path = file.name

    text = load_pdf_text(pdf_path)
    chunks = split_by_sentences(text, 2, 1)

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append(Document(
            page_content=chunk,
            metadata={"chunk_id": i}
        ))

    # ✅ Use documents, not just plain chunks
    db = FAISS.from_documents(documents, embedding=embedding_model)
    return "✅ PDF processed and indexed."




# Step 2: Handle user question
def answer_question(query):
    if not db:
        return "Please upload and process a PDF first."
    docs = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""Answer the question very precisely based only on the context below.

Context:
{context}

Question:
{query}

Answer:"""
    response = llm(prompt)

    sources = "\n\n".join([
        f"🔹 **Chunk {doc.metadata.get('chunk_id', '?')}**:\n{doc.page_content}"
        for doc in docs
    ])

    return response, sources

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("### 🧠 Chat with PDF — Source-aware Answers")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF")
        upload_btn = gr.Button("Process PDF")

    query_input = gr.Textbox(label="Ask a question about the PDF")
    submit_btn = gr.Button("Get Answer")

    answer_output = gr.Textbox(label="Answer")
    source_output = gr.Markdown(label="🔍 Context Sources")

    upload_btn.click(process_pdf, inputs=pdf_input, outputs=answer_output)
    submit_btn.click(answer_question, inputs=query_input, outputs=[answer_output, source_output])

demo.launch(debug=True)
