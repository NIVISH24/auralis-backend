import os
from json import dump
from typing import List

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
from scrapy.utils.project import get_project_settings
from uvicorn import run as run_app

from model import create_rag_pipeline, query_rag_pipeline
from webscraper.spiders.scraper import WebSpider

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_FOLDER = "uploads"
RAG_UPLOADS_FOLDER = "rag"


class RAGRequest(BaseModel):
    model_name: str
    query: str


@app.post("/upload")
async def upload_files(
    model_name: str = Form(...),
    rag_files: List[UploadFile] = File(None),
) -> Response:
    """
    Handles file uploads for RAG and fine-tuning.
    """
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")

    model_folder = os.path.join(UPLOADS_FOLDER, model_name)
    rag_folder = os.path.join(model_folder, RAG_UPLOADS_FOLDER)

    for folder in [model_folder, rag_folder]:
        os.makedirs(folder, exist_ok=True)

    if not rag_files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    def save_file(file: UploadFile, folder: str) -> str:
        file_path = os.path.join(folder, file.filename or "")
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return file.filename or ""

    uploaded_files = {
        "ragFiles": [save_file(file, rag_folder) for file in rag_files]
        if rag_files
        else [],
    }

    create_rag_pipeline(model_name)

    return JSONResponse(
        content={"uploaded_files": uploaded_files, "RAG Pipeline ready": True}
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


class TopicRequest(BaseModel):
    topic: str


def search_duckduckgo(topic: str, max_results: int = 5) -> List[str]:
    """Searches DuckDuckGo and returns a list of URLs."""
    with DDGS() as ddgs:
        search_results = ddgs.text(topic, max_results=max_results)
    return [result["href"] for result in search_results if "href" in result]


def save_links_to_file(links: List[str], filename: str = "urls.json") -> None:
    """Saves URLs to a file for Scrapy to use."""
    with open(filename, "w") as f:
        dump(links, f)


@app.post("/search-and-scrape")
async def search_and_scrape(
    request: TopicRequest,
    background_tasks: BackgroundTasks,
) -> Response:
    """
    Searches DuckDuckGo for a topic, scrapes the top websites, and stores data in ChromaDB.
    """
    topic = request.topic
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    urls = search_duckduckgo(topic, max_results=20)
    if not urls:
        return JSONResponse(
            content={"error": "No relevant URLs found"}, status_code=404
        )

    save_links_to_file(urls, "urls.json")

    process = CrawlerRunner(get_project_settings())
    background_tasks.add_task(process.crawl, WebSpider, urls=urls)

    return JSONResponse(content={"message": "Scraping completed", "urls_scraped": urls})


if __name__ == "__main__":
    run_app(app)
