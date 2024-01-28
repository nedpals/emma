from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from chain import create_handbook_retrieval_chain
from embedding import load_embeddings

import uvicorn

app = FastAPI(
    title="UIC Handbook Assistant API",
    version="0.0.1",
    description="API for the UIC Handbook Assistant",
)

# app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend/dist")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

vector = load_embeddings()
chain = create_handbook_retrieval_chain(vector, history_aware=False)

class InvokeChainRequest(BaseModel):
    config: dict
    input: dict
    kwargs: dict

@app.post("/invoke")
async def invoke_chain(request: InvokeChainRequest):
    result = chain.invoke(request.input, request.config)
    return {
        "answer": result['answer'],
    }

def run_server(host = "localhost", port = 8000):
    uvicorn.run(app, host=host, port=port)
