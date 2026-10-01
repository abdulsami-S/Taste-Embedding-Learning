from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import os

from server.retrieval import load_all, embed_recipe, embed_text, top_k, get_recipe_count
from server.chat import get_intent_and_query, generate_reply

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "file://"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimilarRequest(BaseModel):
    name: str
    ingredients: List[str]
    k: int = 5

class Message(BaseModel):
    role: str
    content: str = Field(..., max_length=500)

class ChatRequest(BaseModel):
    messages: List[Message]
    k: int = 5

@app.get("/health")
async def health():
    return {"status": "ok", "recipes": get_recipe_count()}

@app.post("/similar")
async def similar(req: SimilarRequest):
    try:
        vec = embed_recipe(req.name, req.ingredients)
        results = top_k(vec, req.k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")
        
    messages = [m.model_dump() for m in req.messages[-10:]]
    for msg in messages:
        if msg["role"] not in ["user", "assistant"]:
            raise HTTPException(status_code=400, detail="Invalid role. Must be 'user' or 'assistant'.")
            
    try:
        query_info = get_intent_and_query(messages)
        intent = query_info["intent"]
        
        results = []
        if intent == "recommend":
            name = query_info.get("name")
            ingredients = query_info.get("ingredients")
            if name and ingredients:
                vec = embed_recipe(name, ingredients)
            else:
                vec = embed_text(query_info["query_text"])
            results = top_k(vec, req.k)
            
        reply = generate_reply(intent, messages, results)
        
        return {
            "reply": reply,
            "query": query_info,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error during chat processing")

# Mount client folder as static files
if os.path.isdir("client"):
    app.mount("/", StaticFiles(directory="client", html=True), name="client")
