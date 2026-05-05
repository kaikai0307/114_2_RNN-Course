import requests
import json
import os
from Retrieval import advanced_rag_retrieve, load_vector_db


def query_ollama(prompt, model=None):
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate"
    model = model or os.environ.get("OLLAMA_MODEL", "mistral")
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=data)
        payload = response.json()
        if response.status_code != 200:
            message = payload.get("error") if isinstance(payload, dict) else response.text
            return f"Error calling Ollama ({response.status_code}): {message}"
        if isinstance(payload, dict):
            if "response" in payload:
                return payload["response"]
            if "error" in payload:
                return f"Error calling Ollama: {payload['error']}"
        return f"Error calling Ollama: unexpected response payload: {payload}"
    except Exception as e:
        return f"Error calling Ollama: {e}"

def run_rag_pipeline(query):
    # 1. Retrieve & Rerank
    retrieved_docs = advanced_rag_retrieve(query, vector_db)
    
    # 2. Construct Prompt
    context_text = "\n\n".join([d.page_content for d in retrieved_docs])
    
    prompt = f"""
    <|start_header_id|>system<|end_header_id|>
    You are a helpful science assistant. Answer the question based ONLY on the context provided below.
    If the answer is not in the context, say "I don't know".
    
    Context:
    {context_text}
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Question: {query}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    
    # 3. Generate
    print("\nGenerating Answer...")
    answer = query_ollama(prompt)
    return answer


chunk_size = 1024
if chunk_size == 512:
    vector_db = load_vector_db("db_512/chunks_b", "chunks_b")
elif chunk_size == 1024:
    vector_db = load_vector_db("db_1024/chunks_b", "chunks_b")
else:
    raise ValueError(f"Unsupported chunk_size: {chunk_size}")

# --- Final Execution ---
q1 = "What is the equation for Newton's second law?"
print(f"\nQuestion:{q1}")
answer1 = run_rag_pipeline(q1)
print(f"\nFinal Answer:\n{answer1}")

print("-" * 50)

q2 = "How do plants convert light?"

print(f"\nQuestion:{q2}")
answer2 = run_rag_pipeline(q2)
print(f"\nFinal Answer:\n{answer2}")
