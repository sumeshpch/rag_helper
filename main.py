"""
main.py

Simple Retrieval-Augmented Generation (RAG) example:
- Uses local docs to find context
- Retrieves top matches via embeddings
- Generates answer with OpenAI GPT

Requirements:
    pip install openai numpy sentence-transformers scikit-learn

Usage:
    export OPENAI_API_KEY="your_api_key"
    python rag_helper.py
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from transformers import pipeline

# -------------------------
# CONFIGURATION
# -------------------------
MODEL_EMB = "all-MiniLM-L6-v2"  # Local embedding model
GPT_MODEL = "gpt-4o-mini"       # Fast & affordable GPT model
TOP_K = 3                       # Number of retrieved contexts

# HuggingFace model options (free, no authentication required):
# - "distilgpt2" - Very small, fast, 82MB (basic quality)
# - "gpt2" - Medium, 548MB (better quality, recommended)
# - "gpt2-medium" - Larger, 1.5GB (good quality)
# - "EleutherAI/gpt-neo-125M" - Small, 125M params
# For gated models like Mistral, you need: huggingface-cli login
HUGGINGFACE_MODEL = "gpt2"  # Better quality than distilgpt2

# Set to False to use OpenAI (requires OPENAI_API_KEY)
# Set to True to use local HuggingFace model (no API key needed)
USE_HUGGINGFACE = True

# Initialize clients based on selection
client = None
generator = None

if not USE_HUGGINGFACE:
    # Only initialize OpenAI if we're using it
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found. Set USE_HUGGINGFACE=True to use local model.")
    client = OpenAI(api_key=api_key)

# -------------------------
# SAMPLE KNOWLEDGE BASE
# -------------------------
DOCUMENTS = [
    "Adobe Commerce is built on Magento and provides enterprise-level eCommerce features.",
    "GraphQL in Magento allows efficient data fetching for headless frontends.",
    "A type resolver in GraphQL maps a schema type to its backend data source.",
    "RAG combines data retrieval and generative AI to answer domain-specific questions.",
    "Unit testing ensures the reliability of business logic and critical workflows.",
    "Vector databases store and retrieve high-dimensional embeddings for similarity search.",
    "OpenAI models like GPT-4 can generate human-like text when given contextual information."
]

# -------------------------
# 1. Build embeddings
# -------------------------
print("🔹 Loading embedding model...")
embedder = SentenceTransformer(MODEL_EMB)
doc_embeddings = embedder.encode(DOCUMENTS, normalize_embeddings=True)

# -------------------------
# 2. Retrieve relevant context
# -------------------------
def retrieve_context(query: str, top_k: int = TOP_K):
    """Find top-k relevant docs for the given query."""
    query_emb = embedder.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(query_emb, doc_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    top_docs = [DOCUMENTS[i] for i in top_indices]
    return top_docs

# -------------------------
# 3. Generate final answer
# -------------------------
def generate_answer(query: str, context_docs):
    """Generate an answer using GPT with retrieved context."""
    context = "\n".join([f"- {doc}" for doc in context_docs])
    prompt = f"""
You are an AI assistant helping answer questions using the context below.

Context:
{context}

Question: {query}

Provide a clear and concise answer based only on the context.
"""
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# -------------------------
# 3. Generate final answer with HuggingFace model
# -------------------------
def generate_answer_with_huggingface(query: str, context_docs):
    """Generate an answer using HuggingFace with retrieved context."""
    global generator
    
    # Lazy load the generator only when needed
    if generator is None:
        print(f"🔹 Loading HuggingFace model: {HUGGINGFACE_MODEL}")
        generator = pipeline("text-generation", model=HUGGINGFACE_MODEL, max_length=512)
    
    context = "\n".join([f"- {doc}" for doc in context_docs])
    
    # Simpler prompt that works better with small models
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    response = generator(
        prompt, 
        max_new_tokens=80,  # Shorter to prevent repetition
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,  # Penalize repetition
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    # Extract only the generated answer (remove the prompt)
    full_text = response[0]['generated_text']
    answer = full_text[len(prompt):].strip()
    
    # Clean up: take only the first sentence or paragraph
    # Stop at newlines or repeated patterns
    lines = answer.split('\n')
    clean_answer = lines[0] if lines else answer
    
    # Stop at "Question:" if it appears (model repeating pattern)
    if "Question:" in clean_answer:
        clean_answer = clean_answer.split("Question:")[0].strip()
    
    return clean_answer if clean_answer else "Unable to generate a clear answer with the local model."

# -------------------------
# 4. Example query
# -------------------------
if __name__ == "__main__":
    query = input("Enter your question: ").strip() or "What is RAG and how is it used in Magento?"
    top_docs = retrieve_context(query)
    print("\n🔹 Retrieved Contexts:")
    for d in top_docs:
        print(" -", d)

    print("\n🔹 Generating answer...")
    if USE_HUGGINGFACE:
        answer = generate_answer_with_huggingface(query, top_docs)
    else:
        answer = generate_answer(query, top_docs)
    print("\n💬 Answer:\n", answer)
