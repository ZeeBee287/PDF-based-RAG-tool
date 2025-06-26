import os
import faiss
import PyPDF2
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

def extract_and_chunk_pdf(uploaded_file, chunk_size=300) -> List[str]:
    reader = PyPDF2.PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + " "
    words = full_text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def embed_chunks(chunks: List[str], model_name="sentence-transformers/all-MiniLM-L6-v2") -> Tuple[faiss.IndexFlatL2, List[str]]:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks

def query_with_groq_llama3(query: str, index: faiss.IndexFlatL2, chunks: List[str], model_name="llama3-8b-8192") -> str:
    import openai
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vec = embedder.encode([query]).astype("float32")
    D, I = index.search(query_vec, k=3)
    context = "\n".join([chunks[i] for i in I[0]])

    prompt = f"""Use the context below to answer the question:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""
    openai.api_key = os.getenv("GROQ_API_KEY")
    openai.api_base = "https://api.groq.com/openai/v1"

    if not openai.api_key:
        return "Error: GROQ_API_KEY not set."

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response['choices'][0]['message']['content']
