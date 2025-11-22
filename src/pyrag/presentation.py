from typing import Annotated
from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from dishka.integrations.fastapi import FromDishka, DishkaRoute
import openai
from qdrant_client import QdrantClient
from pyrag.config import Config
from pyrag.infra import LLMRequest, embeddings_to_points, generate_collection, get_embeddings_text, get_from_qdrant, parse_pdf, splitter, upload_to_collection

router = APIRouter(route_class=DishkaRoute, prefix="/api")

@router.post("/create_kb")
async def create_kb(qdrant: FromDishka[QdrantClient], config: FromDishka[Config]) -> str:
    return await generate_collection(qdrant, config)


@router.post("/upload_kb/{collection}")
async def upload_kb(collection: str, 
                    file: Annotated[UploadFile, File()],
                    config: FromDishka[Config],
                    qdrant: FromDishka[QdrantClient],
                    embed:  FromDishka[openai.Client],
                    chunk_size: int =  Body(default = 100),
                    chunk_overlap: int =  Body(default = 25),) -> None:
    embeds = None
    raw_text = None
    if file.content_type == "text/plain":
        raw_text = (await file.read()).decode()
    elif file.content_type == "application/pdf":
        raw_text = await parse_pdf(await file.read())
    else:
        raise HTTPException(400, detail="Unknown content_type")    

    text = await splitter(raw_text, chunk_size = chunk_size, chunk_overlap = chunk_overlap)

    embeds = await get_embeddings_text(
        embed,
        config.embedding.model,
        data = text
    )

    ps = await embeddings_to_points(embeds)

    await upload_to_collection(
        qdrant,
        collection,
        ps,
    )

@router.post("/search_kb/{collection}")
async def search_kb(collection: str, 
                    query: Annotated[list[str], Body()],
                    config: FromDishka[Config],
                    qdrant: FromDishka[QdrantClient],
                    embed:  FromDishka[openai.Client]) -> list[list[str]]:
    q = await get_embeddings_text(embed, config.embedding.model, data = query)

    result = await get_from_qdrant(qdrant, collection, q)
    return result
    
@router.post("/llm/{collection}")
async def search_kb(collection: str, 
                    query: Annotated[list[str], Body()],
                    llm: FromDishka[LLMRequest]) -> str:
    return await llm(query, collection)