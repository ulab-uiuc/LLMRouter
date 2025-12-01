"""
FastAPI backend for LLM-powered documentation search using NVIDIA API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

load_dotenv()

print("\n" + "="*60)
print("Starting LLM Documentation Search API (NVIDIA)")
print("="*60)

app = FastAPI(title="LLM Documentation Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chongshan0lin.github.io/LLMRouterWebsite/",  # Your GitHub Pages URL
        "http://localhost:3000",            # For local testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get NVIDIA API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY
)

# Available NVIDIA models - use these exact names
AVAILABLE_MODELS = {
    "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
}

# Default model
DEFAULT_MODEL = AVAILABLE_MODELS["llama3.1-8b"]

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = chroma_client.get_collection("documentation")
    print(f"✓ Loaded collection with {collection.count()} documents")
except Exception as e:
    print(f"❌ Error loading collection: {e}")
    collection = None

print("="*60 + "\n")

class SearchQuery(BaseModel):
    query: str
    n_results: Optional[int] = 5
    model: Optional[str] = DEFAULT_MODEL
    max_tokens: Optional[int] = 1024

class SearchResponse(BaseModel):
    answer: str
    sources: List[dict]
    model_used: str
    tokens_used: Optional[dict] = None

@app.get("/")
def root():
    return {
        "message": "LLM Documentation Search API",
        "provider": "NVIDIA",
        "status": "running",
        "available_models": AVAILABLE_MODELS,
        "default_model": DEFAULT_MODEL,
        "documents": collection.count() if collection else 0
    }

@app.get("/health")
def health():
    return {"status": "healthy", "documents": collection.count() if collection else 0}

@app.get("/models")
def list_models():
    """List available models"""
    return {
        "models": [
            {
                "id": model_id,
                "name": model_name,
                "provider": "NVIDIA"
            }
            for model_id, model_name in AVAILABLE_MODELS.items()
        ],
        "default": DEFAULT_MODEL
    }

@app.post("/search", response_model=SearchResponse)
async def search_documentation(query: SearchQuery):
    """Search documentation using NVIDIA's LLM API"""
    
    print("\n" + "="*60)
    print(f"📥 Query: {query.query[:50]}...")
    print(f"🤖 Model: {query.model}")
    print("="*60)
    
    if not collection:
        raise HTTPException(status_code=503, detail="Collection not loaded")
    
    try:
        # Search documents
        results = collection.query(
            query_texts=[query.query],
            n_results=query.n_results
        )
        
        if not results['documents'][0]:
            return SearchResponse(
                answer="No relevant information found.",
                sources=[],
                model_used=query.model
            )
        
        print(f"✓ Found {len(results['documents'][0])} documents")
        
        # Build context
        context_parts = []
        sources = []
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            # Limit document length to avoid token limits
            doc_preview = doc[:500] if len(doc) > 500 else doc
            context_parts.append(f"[Source {i+1}]:\n{doc_preview}")
            sources.append({
                'title': metadata.get('title', 'Unknown'),
                'source': metadata.get('source', ''),
                'url': f"/{metadata.get('source', '').replace('.md', '/')}"
            })
        
        context = "\n\n".join(context_parts)
        
        # Simple, direct prompt
        prompt = f"""Based on the following documentation excerpts, answer the user's question.

Documentation:
{context}

Question: {query.query}

Provide a clear, concise answer based only on the information above."""

        # Call NVIDIA API
        print("🤖 Calling NVIDIA API...")
        
        response = client.chat.completions.create(
            model=query.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=query.max_tokens,
            temperature=0.5,
            top_p=1.0,
            stream=False
        )
        
        answer = response.choices[0].message.content
        
        tokens_used = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        print(f"✅ Success! Tokens: {tokens_used['total_tokens']}")
        
        return SearchResponse(
            answer=answer,
            sources=sources,
            model_used=query.model,
            tokens_used=tokens_used
        )
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)