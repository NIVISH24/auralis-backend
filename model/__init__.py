from pathlib import Path
from typing import Any, Dict, List, Tuple

from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from torch.cuda import is_available as is_cuda_available

ollama = Ollama(model="deepseek-r1")
device = "cuda" if is_cuda_available() else "cpu"
embed_model_prompt = HuggingFaceEmbeddings(
    model_name="dunzhang/stella_en_400M_v5",
    model_kwargs={
        "trust_remote_code": True,
        "device": "cpu",
        "config_kwargs": {
            "use_memory_efficient_attention": False,
            "unpad_inputs": False,
        },
    }
    if device == "cpu"
    else {
        "trust_remote_code": True,
        "device": "cuda",
    },
    encode_kwargs={
        "prompt_name": "s2p_query",
    },
)
adapter_embed_prompt = LangchainEmbedding(embed_model_prompt)


class EmbeddingFunctionPrompt(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [
            embedding
            for embedding in adapter_embed_prompt.get_text_embedding(input)  # type: ignore
        ]


def create_rag_pipeline(
    model_name: str,
    scraped_data: List[Dict[str, Any]],
) -> Tuple[Dict[str, str], int]:
    base_path = Path.cwd() / "uploads" / model_name
    rag_path = base_path / "rag"
    db_path = base_path / "chroma.sqlite3"

    if db_path.exists():
        return {
            "message": "RAG pipeline already created",
            "embeddings_db_path": db_path.as_posix(),
        }, 200
    elif not rag_path.exists() and not scraped_data:
        return {"error": "No files uploaded or data scraped"}, 400

    # Add scraped data to documents

    db = PersistentClient(base_path.as_posix())
    collection = db.create_collection(
        "embeddings_db",
        embedding_function=EmbeddingFunctionPrompt(),
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    if any(rag_path.iterdir()):
        documents = SimpleDirectoryReader(
            rag_path,
            recursive=True,
            exclude_hidden=True,
            errors="ignore",
        ).load_data(show_progress=True)
        VectorStoreIndex.from_documents(
            documents,
            storage_context,
            embed_model=adapter_embed_prompt,
            show_progress=True,
        )
        for data in scraped_data:
            documents.append(Document(**data))

    return {
        "message": "RAG pipeline created and embeddings saved",
        "embeddings_db_path": db_path.as_posix(),
    }, 200


def query_rag_pipeline(model_name: str, query: str) -> Tuple[Dict[str, Any], int]:
    base_path = Path.cwd() / "uploads" / model_name
    db_path = base_path / "chroma.sqlite3"

    if not db_path.exists():
        return {"error": "RAG pipeline not created"}, 400

    db = PersistentClient(base_path.as_posix())
    collection = db.get_collection("embeddings_db")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=adapter_embed_prompt,
    )
    query_engine = index.as_query_engine(llm=ollama)
    response = query_engine.query(query)

    if response:
        return response.response.model_dump(), 200  # type: ignore

    return {"error": "No response found"}, 404
