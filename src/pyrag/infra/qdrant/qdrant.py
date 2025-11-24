import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from qdrant_client.conversions.common_types import QueryResponse

from pyrag.config import QdrantConfig
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
                    size=self.config.dim,
                    distance=Distance.COSINE,
                ),
            },
        )

        if not res:
            raise RuntimeError("Something went wrong creating collection")  # noqa

        return name

    async def upload_to_collection(
        self,
        collection_name: str,
        data: list[PointStruct],
    ) -> None:
        self.client.upload_points(
            collection_name=collection_name,
            points=data,
        )

    async def get_from_qdrant(
        self,
        collection_name: str,
        data: list[Embedding],
    ) -> list[dict]:
        SCORE_FIELD = "__score"

        result: list[dict] = []
        for embedding in data:

            response: QueryResponse = self.client.query_points(
                collection_name=collection_name,
                query=embedding.vector,
                using="cvector",
            )

            for point in response.points:
                payload = point.payload
                payload[SCORE_FIELD] = point.score

                result.append(payload)
        result.sort(
            key = lambda i: i[SCORE_FIELD],
            reverse = True
        )
        return result
