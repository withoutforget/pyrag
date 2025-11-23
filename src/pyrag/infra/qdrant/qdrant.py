from dataclasses import dataclass
import io
import json
import logging
import uuid
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from pyrag.config import Config, QdrantConfig

from PyPDF2 import  PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pyrag.infra.openai.embedder import Embedding

@dataclass(slots=True)
class Qdrant:
    config: QdrantConfig
    client: QdrantClient

    async def generate_random_collection(self) -> str:
        name = uuid.uuid4().hex

        res = self.client.create_collection(
            name,
            vectors_config={
                "cvector": VectorParams(
                    size = self.config.dim,
                    distance = Distance.COSINE,
                )
            }
        )

        if res != True:
            raise RuntimeError("Something went wrong creating collection")

        return name

    async def upload_to_collection(self, collection_name: str, data: list[PointStruct]) -> None:
        self.client.upload_points(
            collection_name=collection_name,
            points=data,
        )    

    async def get_from_qdrant(
            self,
        collection_name: str,
        data: list[Embedding]
    ) -> list[list[str]]:
        # TODO: fix this one return value to list[dict]
        result: list[list[str]] = []
        for slice, _ in data:
            response = self.client.query_points(
                collection_name=collection_name,
                query = slice,
                using="cvector"
            )
            res: list[str] = [p.payload.get("text")
                            for p in response.points 
                            if p.payload is not None]
            result.append(list(filter(lambda p: p is not None, res)))
        return result
