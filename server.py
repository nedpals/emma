import json
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import starlette.status as status
import uvicorn

import meta
from agent_setup import create_agent
from models import Message

# App environment
environment = os.environ.get("ENV", "development")
manifest = {}

if environment != "development":
    with open("public/.vite/manifest.json") as f:
        manifest = json.load(f)

agent = create_agent()

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
    input_text = request.input.input
    chat_history = [m.model_dump() for m in request.input.chat_history] if request.input.chat_history else []

    async def event_generator():
        async for event in agent.run(input_text, chat_history):
            yield {"event": event["type"], "data": json.dumps(event)}

    return EventSourceResponse(event_generator())


def run_server(host="localhost", port=8000):
    uvicorn.run(app, host=host, port=port)
