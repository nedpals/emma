from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from langserve import add_routes
from chain import create_handbook_retrieval_chain
from embedding import load_embeddings

import uvicorn

def create_server():
    app = FastAPI(
        title="UIC Handbook Assistant API",
        version="0.0.1",
        description="API for the UIC Handbook Assistant",
    )

    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend/dist")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
    )

    vector = load_embeddings()

    add_routes(
        app,
        create_handbook_retrieval_chain(vector),
    )

    return app

def run_server(host = "localhost", port = 8000):
    app = create_server()
    uvicorn.run(app, host=host, port=port)
