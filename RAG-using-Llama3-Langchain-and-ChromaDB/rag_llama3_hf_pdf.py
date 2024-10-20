from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import faiss
import numpy as np
import gradio as gr

import nltk

# Download the required NLTK data
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

# Load Sentence Transformer model
sentence_transformer_model_id = "mixedbread-ai/mxbai-embed-large-v1"
ST = SentenceTransformer(sentence_transformer_model_id)

# Function to read and extract text from PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to split text into chunks with complete sentences
def create_text_chunks(text, min_words=100, max_words=200):
    
    sentences = sent_tokenize(text) 
    
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Check if adding the next sentence would exceed the max_words limit
        if len(current_chunk.split()) + len(sentence.split()) > max_words:
            # If so, save the current chunk and start a new one
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            # Otherwise, add the sentence to the current chunk
            current_chunk += " " + sentence
        
        # If the current chunk is within the minimum length, add it to the chunks
        if len(current_chunk.split()) >= min_words and len(current_chunk.split()) <= max_words:
            chunks.append(current_chunk.strip())
            current_chunk = ""

    # Add any remaining text as the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Create overlapping chunks to preserve context
    overlapping_chunks = []
    overlap_size = 2  # Number of overlapping sentences
    for i in range(len(chunks)):
        overlapping_chunk = chunks[i]
        if i < len(chunks) - 1:
            overlap_sentences = sent_tokenize(chunks[i])[-overlap_size:]
            overlapping_chunk += " " + " ".join(overlap_sentences)
        overlapping_chunks.append(overlapping_chunk.strip())

    return overlapping_chunks

# Function to embed the extracted text
def embed_text(text_chunks):
    embeddings = ST.encode(text_chunks, convert_to_tensor=True)
    return embeddings, text_chunks

# Provide the path to the PDF file
pdf_path = "/home/auishik/RAG-using-Llama3-Langchain-and-ChromaDB/M-618.pdf"  # Replace with your actual PDF path
text = extract_text_from_pdf(pdf_path)


# Generate text chunks
text_chunks = create_text_chunks(text, min_words=100, max_words=200)
# Print each text chunk on a new line
# for i, chunk in enumerate(text_chunks):
#     print(f"Chunk {i+1}:\n{chunk}\n")


# Embed the extracted text
embeddings, text_chunks = embed_text(text_chunks)


# Create a FAISS index for searching
dimension = embeddings.shape[1]  # Dimension of the embeddings
print(dimension)

index = faiss.IndexFlatL2(dimension)  # L2 distance index
print(index)
index.add(embeddings.cpu().numpy())
print(index)


# Define the search function
def search(query: str, k: int = 3):
    """Embed a query and return the most similar results."""
    embedded_query = ST.encode([query], convert_to_tensor=True)
    distances, indices = index.search(embedded_query.cpu().numpy(), k)
    retrieved_texts = [text_chunks[i] for i in indices[0]]  # Retrieve the relevant text chunks
    return distances, retrieved_texts

# Configure and load the LLaMA model with BitsAndBytes for 4-bit quantization
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config
)

# Define terminators and system prompt
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer."""

def format_prompt(prompt, retrieved_texts, k):
    """Format the prompt using retrieved texts for model input."""
    PROMPT = f"Question: {prompt}\nContext:"
    for idx in range(min(k, len(retrieved_texts))):
        PROMPT += f"{retrieved_texts[idx]}\n"
    return PROMPT

def generate(formatted_prompt):
    """Generate a response using the language model."""
    formatted_prompt = formatted_prompt[:2000]  # Truncate to avoid GPU OOM

    # Combine system prompt and user content into a single string
    prompt_input = f"{SYS_PROMPT}\n{formatted_prompt}"
    
    # Tokenize the input
    input_ids = tokenizer(prompt_input, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        input_ids['input_ids'],
        max_new_tokens=1024,
        eos_token_id=terminators[0],  # Ensure eos_token_id is correctly used
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    # Decode the generated response
    response = outputs[0][input_ids['input_ids'].shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    print(response)
    return response


def rag_chatbot(prompt: str, k: int = 2):
    """Retrieve documents, format the prompt, and generate a response."""
    print(prompt)
    scores, retrieved_texts = search(prompt, k)
    print(scores, retrieved_texts)
    exit()
    formatted_prompt = format_prompt(prompt, retrieved_texts, k)
    return generate(formatted_prompt)

if __name__ == '__main__':
    iface = gr.Interface(
        fn=rag_chatbot,  # Function to wrap
        inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
        outputs=gr.Textbox(),
        title="Document Question Answering",
        description="Ask questions about the document and get answers based on the content.",
    )
    iface.launch()

# Test the chatbot function
print(rag_chatbot("How can I learn english there?", k=2))
