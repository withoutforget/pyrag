from typing import Annotated

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, Body, File, HTTPException, UploadFile

from pyrag.infra.openai.embedder import Content, Embedder
from pyrag.infra.openai.llm import LLMRequest
from pyrag.infra.parsers.pdf import PDFParser
from pyrag.infra.parsers.splitter import TextSplitter
from pyrag.infra.qdrant.qdrant import Qdrant

ROUTER = APIRouter(route_class=DishkaRoute)


@ROUTER.post("/create")
async def create_kb(qdrant: FromDishka[Qdrant]) -> str:
    return await qdrant.generate_random_collection()


@ROUTER.post("/upload/{collection}")
async def upload_kb(
    collection: str,
    file: Annotated[UploadFile, File()],
    qdrant: FromDishka[Qdrant],
    pdf_parser: FromDishka[PDFParser],
    text_splitter: FromDishka[TextSplitter],
    embedder: FromDishka[Embedder],
    chunk_size: int = Body(default=100),  # noqa
    chunk_overlap: int = Body(default=25),  # noqa
) -> None:
    raw_text = None
    if file.content_type == "text/plain":
        raw_text = (await file.read()).decode()
    elif file.content_type == "application/pdf":
        raw_text = await pdf_parser.parse(await file.read())
    else:
        raise HTTPException(400, detail="Unknown content_type")

    text = await text_splitter(
        raw_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    ## maybe we want to standartize this payload?

    embeddings = await embedder.get_embeddings_text(
        [
            Content(
                text=t,
                payload={
                    "document": file.filename,
                    "content_type": file.content_type,
                    "text": t,
                },
            )
            for t in text
        ],
    )

    await qdrant.upload_to_collection(
        collection,
        data=[e.to_point_struct() for e in embeddings],
    )


@ROUTER.post("/search/{collection}")
async def search_kb(
    collection: str,
    query: Annotated[list[str], Body()],
    qdrant: FromDishka[Qdrant],
    embedder: FromDishka[Embedder],
) -> list[dict]:
    q = await embedder.get_embeddings_text(
        data=[Content(text=t, payload={}) for t in query],
    )
    return await qdrant.get_from_qdrant(collection, q)
