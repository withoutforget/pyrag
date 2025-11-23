from typing import Annotated
from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from dishka.integrations.fastapi import FromDishka, DishkaRoute
import openai
from qdrant_client import QdrantClient
from pyrag.config import Config
from pyrag.infra.openai.embedder import Embedder
from pyrag.infra.openai.llm import LLMRequest
from pyrag.infra.parsers.pdf import PDFParser
from pyrag.infra.qdrant.qdrant import Qdrant

router = APIRouter(route_class=DishkaRoute, prefix="/api")

@router.post("/create_kb")
async def create_kb(qdrant: FromDishka[Qdrant]) -> str:
    return await qdrant.generate_random_collection()


@router.post("/upload_kb/{collection}")
async def upload_kb(collection: str, 
                    file: Annotated[UploadFile, File()],
                    qdrant: FromDishka[Qdrant],
                    pdf_parser: FromDishka[PDFParser],
                    embedder:  FromDishka[Embedder],
                    chunk_size: int =  Body(default = 100),
                    chunk_overlap: int =  Body(default = 25),) -> None:
    raw_text = None
    if file.content_type == "text/plain":
        raw_text = (await file.read()).decode()
    elif file.content_type == "application/pdf":
        raw_text = await pdf_parser.parse(await file.read())
    else:
        raise HTTPException(400, detail="Unknown content_type")    

    text = await parsers(raw_text, chunk_size = chunk_size, chunk_overlap = chunk_overlap)

    embeddings = await embedder.get_embeddings_text(text)

    await qdrant.upload_to_collection(
        collection,
        data = [e.toPointStruct() for e in embeddings]
    )

@router.post("/search_kb/{collection}")
async def search_kb(collection: str, 
                    query: Annotated[list[str], Body()],
                    qdrant: FromDishka[Qdrant],
                    embedder:  FromDishka[Embedder]
        ) -> list[dict]:
    q = await embedder.get_embeddings_text(data = query)
    result = await qdrant.get_from_qdrant(collection, q)
    return result
    
@router.post("/llm/{collection}")
async def search_kb(collection: str, 
                    query: Annotated[list[str], Body()],
                    llm: FromDishka[LLMRequest],
                     qdrant: FromDishka[Qdrant],
                    embedder:  FromDishka[Embedder]) -> str:
    q = await embedder.get_embeddings_text(data = query)
    result = await qdrant.get_from_qdrant(collection, q)
    return await llm.ask(text = query, rag = result)