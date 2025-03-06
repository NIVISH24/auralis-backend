from __future__ import annotations

import os
from asyncio import Future, get_running_loop
from typing import List, Optional

from duckduckgo_search import DDGS
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from twisted.internet.asyncioreactor import install
from twisted.internet.defer import Deferred

from model import create_rag_pipeline, query_rag_pipeline
from webscraper.spiders.scraper import WebSpider

install()

configure_logging()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
runner = CrawlerRunner(get_project_settings())

UPLOADS_FOLDER = "uploads"
RAG_UPLOADS_FOLDER = "rag"


def deferred_to_future(d: Deferred) -> Future:
    loop = get_running_loop()
    future = loop.create_future()

    def callback(result):
        if not future.cancelled():
            future.set_result(result)
        return result

    def errback(failure):
        if not future.cancelled():
            future.set_exception(failure.value)
        return failure

    d.addCallbacks(callback, errback)
    return future


class RAGRequest(BaseModel):
    model_name: str
    query: str


class TopicRequest(BaseModel):
    topic: str


def search_duckduckgo(topic: str, max_results: int = 5) -> List[str]:
    """Searches DuckDuckGo and returns a list of URLs."""
    with DDGS() as ddgs:
        search_results = ddgs.text(topic, max_results=max_results)
    return [result["href"] for result in search_results if "href" in result]


@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
    topic: Optional[str] = Form(None),
    urls: Optional[List[str]] = Form(None),
    rag_files: Optional[List[UploadFile]] = File(None),
    text_inputs: Optional[List[str]] = Form(None),
) -> Response:
    """
    Handles file uploads for RAG and fine-tuning, and topic-based scraping.
    """
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")

    model_folder = os.path.join(UPLOADS_FOLDER, model_name)
    rag_folder = os.path.join(model_folder, RAG_UPLOADS_FOLDER)

    for folder in [model_folder, rag_folder]:
        os.makedirs(folder, exist_ok=True)

    uploaded_files = []

    if rag_files:

        def save_file(file: UploadFile, folder: str) -> str:
            file_path = os.path.join(folder, file.filename or "")
            with open(file_path, "wb") as f:
                f.write(file.file.read())
            return file.filename or ""

        uploaded_files = [save_file(file, rag_folder) for file in rag_files]

    scraped_data = []

    if topic and not urls:
        urls = search_duckduckgo(topic, max_results=20)
        if not urls:
            return JSONResponse(
                content={"error": "No relevant URLs found"}, status_code=404
            )

    if urls:
        await deferred_to_future(runner.crawl(WebSpider, urls=urls, name=model_name))

    if text_inputs:
        for text in text_inputs:
            scraped_data.append({"content": text})

    create_rag_pipeline(model_name, scraped_data)

    return JSONResponse(
        content={
            "uploaded_files": uploaded_files,
            "RAG Pipeline ready": True,
            "message": "Scraping completed" if topic or urls else "Files uploaded",
        }
    )


@app.get("/models")
async def get_models() -> Response:
    """
    Lists all models by checking the upload directory for folders.
    """
    if not os.path.exists(UPLOADS_FOLDER):
        os.makedirs(UPLOADS_FOLDER)

    models = [
        model_name
        for model_name in os.listdir(UPLOADS_FOLDER)
        if os.path.isdir(os.path.join(UPLOADS_FOLDER, model_name))
    ]

    return JSONResponse(content={"models": models})


@app.post("/rag")
async def rag_algorithm(request: RAGRequest) -> Response:
    """
    Runs the RAG algorithm on the specified model and query.
    """
    model_name = request.model_name
    query = request.query

    try:
        result, status_code = query_rag_pipeline(model_name, query)
        return JSONResponse(content={"results": result}, status_code=status_code)
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
