from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from chain import create_handbook_retrieval_chain
from embedding import load_embeddings
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage, AIMessage
import starlette.status as status

import json
import meta
import os
import uvicorn

# App environment
environment = os.environ.get("ENV", "development")
manifest = {}

# vite manifest
if environment != "development":
    with open("public/.vite/manifest.json") as f:
        manifest = json.load(f)

vector = load_embeddings()
chain = create_handbook_retrieval_chain(vector, history_aware=True)

app = FastAPI(
    title=f"{meta.title} API",
    version=meta.version,
    description=f"API for the {meta.title} chatbot",
)

templates = Jinja2Templates(directory="templates")

if environment == "development":
    app.mount("/public", StaticFiles(directory="frontend/public", html=True), name="public")
    app.mount("/src", StaticFiles(directory="frontend/src", html=True), name="frontend/src")
else:
    app.mount("/public", StaticFiles(directory="public", html=True), name="public")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

class InvokeChainRequest(BaseModel):
    config: dict
    input: dict
    kwargs: dict

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", context={
        "title": meta.title,
        "description": meta.description,
        "env": environment,
        "manifest": manifest,
    })

@app.get("/{full_path}")
async def catch_all(request: Request, full_path: str):
    return RedirectResponse(url="/public/" + full_path, status_code=status.HTTP_302_FOUND)

@app.post("/invoke")
async def invoke_chain(request: InvokeChainRequest):
    chat_history = []

    if "history" in request.input:
        for msg in request.input["history"]:
            if msg["type"] == "human":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                chat_history.append(AIMessage(content=msg["content"]))

        request.input["chat_history"] = chat_history

        # delete history from input so it doesn't get passed to the chain
        del request.input["history"]

    result = chain.invoke(request.input, request.config)
    return {
        "answer": result['answer'],
    }

def run_server(host = "localhost", port = 8000):
    uvicorn.run(app, host=host, port=port)
