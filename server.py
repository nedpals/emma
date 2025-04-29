from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from prompt import RetrievalChain
from embedding import load_vector_store
from fastapi.templating import Jinja2Templates
from models import Message
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

chain = RetrievalChain(load_vector_store(), history_aware=True)

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

class InvokeChainInput(BaseModel):
    input: str
    chat_history: list[Message] | None = None
    n_results: int = 10

class InvokeChainRequest(BaseModel):
    config: dict
    input: InvokeChainInput
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
    result = chain.invoke(dict(request.input), request.config)
    return result

def run_server(host = "localhost", port = 8000):
    uvicorn.run(app, host=host, port=port)
