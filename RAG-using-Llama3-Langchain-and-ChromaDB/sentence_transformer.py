from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
print(model)

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
print(sentences)

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings)

print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
# similarities = model.similarity(embeddings, embeddings)
# Compute cosine similarity
similarities = cosine_similarity(embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])