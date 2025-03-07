from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from json import loads

from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from ollama import AsyncClient
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from torch.cuda import is_available as is_cuda_available

# Initialize your LLM and embedding model
ollama = Ollama(model="llama3.2", request_timeout=5000.0)
device = "cuda" if is_cuda_available() else "cpu"
embed_model_prompt = HuggingFaceEmbeddings(
    model_name="dunzhang/stella_en_400M_v5",
    model_kwargs=(
        {
            "trust_remote_code": True,
            "device": "cpu",
            "config_kwargs": {
                "use_memory_efficient_attention": False,
                "unpad_inputs": False,
            },
        }
        if device == "cpu"
        else {"trust_remote_code": True, "device": "cuda"}
    ),
    encode_kwargs={"prompt_name": "s2p_query"},
)
adapter_embed_prompt = LangchainEmbedding(embed_model_prompt)


class EmbeddingFunctionPrompt(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # Make sure to convert embeddings to list if needed
        return [
            embedding for embedding in adapter_embed_prompt.get_text_embedding(input)
        ]  # type: ignore


def create_rag_pipeline(
    model_name: str,
    input_data: List[Union[Dict[str, Any], Document]],
) -> Tuple[Dict[str, str], int]:
    """
    Creates a RAG pipeline by combining documents loaded from files (if available)
    with additional input data, which can be either a list of dicts (scraped data)
    or Document objects (e.g. coming from FastAPI).

    Returns a tuple containing a response dictionary and HTTP status code.
    """
    base_path = Path.cwd() / "uploads" / model_name
    rag_path = base_path / "rag"
    db_path = base_path / "chroma.sqlite3"

    # If the pipeline already exists, return immediately.
    if db_path.exists():
        print(f"{rag_path = }")
        return {
            "message": "RAG pipeline already created",
            "embeddings_db_path": db_path.as_posix(),
        }, 200
    elif (not rag_path.exists() or not any(rag_path.iterdir())) and not input_data:
        return {"error": "No files uploaded or data scraped"}, 400

    # Initialize persistent client and collection
    db = PersistentClient(base_path.as_posix())
    collection = db.create_collection(
        "embeddings_db",
        embedding_function=EmbeddingFunctionPrompt(),
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load existing documents from the rag directory, if available
    documents: List[Document] = []
    print(documents)
    if rag_path.exists() and any(rag_path.iterdir()):
        print(f"{True = }")
        documents = SimpleDirectoryReader(
            rag_path,
            recursive=True,
            exclude_hidden=True,
            errors="ignore",
        ).load_data(show_progress=True)
        # Create an index with the loaded documents
        VectorStoreIndex.from_documents(
            documents,
            storage_context,
            embed_model=adapter_embed_prompt,
            show_progress=True,
        )
    print(f"{'*' * 40}")
    # Process the input data: convert dicts to Documents with proper fields.
    for data in input_data:
        print(f"{data = }")
        if isinstance(data, dict):
            # Use the scraped "content" as the main text,
            # and store other fields in metadata.
            text = data.get("content", "")
            metadata = {"url": data.get("url"), "title": data.get("title")}
            print(f"{metadata = }")
            documents.append(Document(text=text, metadata=metadata))
        elif isinstance(data, Document):
            documents.append(data)
        else:
            raise ValueError(
                "Each item in input_data must be a dict or a Document instance."
            )

    # (Re)create the index with all documents (loaded + input)
    VectorStoreIndex.from_documents(
        documents,
        storage_context,
        embed_model=adapter_embed_prompt,
        show_progress=True,
    )

    return {
        "message": "RAG pipeline created and embeddings saved",
        "embeddings_db_path": db_path.as_posix(),
    }, 200


async def stream_chat(message: str) -> str:
    total_parts = ""

    async for part in await AsyncClient().chat(
        model="llama3.2",
        stream=True,
        format="json",
        messages=[{"role": "user", "content": message}],
    ):
        total_parts += part["message"]["content"]

    return total_parts


async def query_rag_pipeline(model_name: str, query: str) -> Tuple[Dict[str, Any], int]:
    """
    Queries the RAG pipeline for the specified model and query string.

    Returns a tuple containing the query result (or error) and an HTTP status code.
    """
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
        response = await stream_chat(splitter_prompt(response.response))
        # Return the result; adjust this dependingdd on your LLM response format.
        return loads(response), 200  # type: ignore

    return {"error": "No response found"}, 404


def splitter_prompt(text: str) -> List[str]:
    """
    Splits the input text into multiple prompts.
    """
    format = {
        "splitted_text": {
            "topic1": "information on topic1 as markdown",
            "topic2": "information on topic2 as markdown",
            "topic3": "information on topic3 as markdown",
        },
    }
    return (
        f"This is the response: {text}\n\n Split it to break the parts of the response wherever"
        "you think is a different topic that can be explored/expanded even more. If it can't be split, give just one element in the list. Give it in json format like in"
        "the example below.:"
        f"{format}"
    )
