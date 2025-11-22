import asyncio
import logging
import uuid
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, QueryRequest

from pyrag.config import get_config

async def get_embeddings_text(
        embed: openai.Client,
        model: str,
        data: list[str]
) -> list[tuple[list[float], str]]:
    result = embed.embeddings.create(
        model = model,
        input = data
    )

    return [(embedddings.embedding, text) for embedddings, text in zip(result.data, data)]

async def embeddings_to_points(embds: tuple[list[float], str]) -> list[PointStruct]:
    return [
        PointStruct(
            id = uuid.uuid4(),
            vector = { "cvector":  v},
            payload = { "text": s }
        )
        for v, s in embds
    ]

async def generate_collection(qdrant: QdrantClient) -> str:
    name = str(uuid.uuid4())

    res = qdrant.create_collection(
        name,
        vectors_config={
            "cvector": VectorParams(
                size = 768,
                distance = Distance.COSINE,
            )
        }
    )
    
    if res != True:
        raise RuntimeError("Something went wrong when creating collection in Qdrant...")
    return name


async def upload_to_collection(qdrant: QdrantClient, 
                               collection_name: str,
                               data: list[PointStruct]):
    qdrant.upload_points(
        collection_name=collection_name,
        points=data,
    )

async def get_from_qdrant(
    qdrant: QdrantClient,
    collection_name: str,
    data: list[tuple[list[float], str]]
) -> list[list[str]]:
    result: list[list[str]] = []
    for slice, _ in data:
        response = qdrant.query_points(
            collection_name=collection_name,
            query = slice,
            using="cvector"
        )
        res: list[str] = [p.payload.get("text")
                           for p in response.points 
                           if p.payload is not None]
        result.append(list(filter(lambda p: p is not None, res)))
    return result

from PyPDF2 import  PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


async def parse_pdf(path: str) -> list[str]:
    pdf = PdfReader(open(path, "rb"))
    text = '\n'.join(t.extract_text() for t in pdf.pages)

    text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n'],
            chunk_size=100,
            chunk_overlap=25,
            length_function=len,
        )
    
    return text_splitter.split_text(text)

async def main():
    config = get_config()

    model: str = config.embedding.model
    
    embed = openai.Client(
        base_url = config.embedding.base_url,
        api_key = config.embedding.api_key,
    )
    
    qdrant = QdrantClient(
        host = config.qdrant.host,
        port = config.qdrant.port,
    )
    collection: str
    data = await parse_pdf("./test.pdf")
    
    embeds = await get_embeddings_text(
        embed,
        model,
        data = data
    )

    collection = await generate_collection(qdrant)

    ps = await embeddings_to_points(embeds)
    await upload_to_collection(
        qdrant,
        collection,
        ps,
    )

    q = await get_embeddings_text(embed, model, data = ["цель работы"])

    result = await get_from_qdrant(qdrant, collection, q)

    logging.info("got result: %s", result)

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
    )

    logging.info("all set up. starting...")
    asyncio.run(main())