from huggingface_hub import notebook_login
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Load Sentence Transformer model
sentence_transformer_model_id = "mixedbread-ai/mxbai-embed-large-v1"
ST = SentenceTransformer(sentence_transformer_model_id)

# Load dataset and embed it using Sentence Transformer
dataset = load_dataset("not-lain/wikipedia")


def embed(batch):
    information = batch["text"]
    print(information)
    return {"embeddings": ST.encode(information)}

dataset = dataset.map(embed, batched=True, batch_size=16)

# Reload dataset with embedded revision
dataset = load_dataset("not-lain/wikipedia", revision="embedded")
data = dataset["train"]

data = data.add_faiss_index("embeddings")  # Add FAISS index for searching

def search(query: str, k: int = 3):
    """Embed a query and return the most similar results."""
    embedded_query = ST.encode(query)
    scores, retrieved_examples = data.get_nearest_examples("embeddings", embedded_query, k=k)
    return scores, retrieved_examples

# Configure and load the LLaMA model with BitsAndBytes for 4-bit quantization
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # Change to float16
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

def format_prompt(prompt, retrieved_documents, k):
    """Format the prompt using retrieved documents for model input."""
    PROMPT = f"Question: {prompt}\nContext:"
    for idx in range(k):
        PROMPT += f"{retrieved_documents['text'][idx]}\n"
    return PROMPT

def generate(formatted_prompt):
    """Generate a response using the language model."""
    formatted_prompt = formatted_prompt[:2000]  # Truncate to avoid GPU OOM
    messages = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": formatted_prompt}]
    
    # Prepare input IDs for model generation
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def rag_chatbot(prompt: str, k: int = 2):
    """Retrieve documents, format the prompt, and generate a response."""
    scores, retrieved_documents = search(prompt, k)
    formatted_prompt = format_prompt(prompt, retrieved_documents, k)
    return generate(formatted_prompt)

# Test the chatbot function
print(rag_chatbot("what's anarchy?", k=2))
