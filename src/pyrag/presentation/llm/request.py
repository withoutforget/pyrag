from typing import Annotated

from dishka.integrations.fastapi import DishkaRoute, FromDishka
from fastapi import APIRouter, Body, File, HTTPException, UploadFile

from pyrag.infra.openai.embedder import Content, Embedder
from pyrag.infra.openai.llm import LLMRequest
from pyrag.infra.parsers.pdf import PDFParser
from pyrag.infra.parsers.splitter import TextSplitter
from pyrag.infra.qdrant.qdrant import Qdrant

ROUTER = APIRouter(route_class=DishkaRoute)


@ROUTER.post("/{collection}")
async def request_llm(
    collection: str,
    query: Annotated[list[str], Body()],
    llm: FromDishka[LLMRequest],
    qdrant: FromDishka[Qdrant],
    embedder: FromDishka[Embedder],
) -> str:
    q = await embedder.get_embeddings_text(
        data=[Content(text=t, payload={}) for t in query],
    )
    result = await qdrant.get_from_qdrant(collection, q)
    return await llm.ask(text=query, rag=result)
