from typing import Annotated
from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from dishka.integrations.fastapi import FromDishka, DishkaRoute
import openai
from qdrant_client import QdrantClient
from pyrag.config import Config
from pyrag.infra import embeddings_to_points, generate_collection, get_embeddings_text, get_from_qdrant, parse_pdf, upload_to_collection

router = APIRouter(route_class=DishkaRoute, prefix="/api")

@router.post("/create_kb")
async def create_kb(qdrant: FromDishka[QdrantClient]) -> str:
    return await generate_collection(qdrant)


@router.post("/upload_kb/{collection}")
async def upload_kb(collection: str, 
                    file: Annotated[UploadFile, File()],
                    config: FromDishka[Config],
                    qdrant: FromDishka[QdrantClient],
                    embed:  FromDishka[openai.Client]) -> None:
    embeds = None
    if file.content_type == "text/plain":
        embeds = await get_embeddings_text(
            embed,
            config.embedding.model,
            data = (await file.read()).decode()
        )
    elif file.content_type == "application/pdf":
        embeds = await get_embeddings_text(
            embed,
            config.embedding.model,
            data = await parse_pdf(await file.read())
        )
    else:
        raise HTTPException(400, detail="Unknown content_type")    

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
    