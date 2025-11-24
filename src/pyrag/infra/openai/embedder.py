from typing import Optional
import uuid
from dataclasses import dataclass

from adaptix import Retort
import openai
from qdrant_client.models import PointStruct


@dataclass(slots=True)
class Embedding:
    vector: list[float]
    payload: dict

    def to_point_struct(self) -> PointStruct:
        return PointStruct(
            id=uuid.uuid4(),
            vector={"cvector": self.vector},
            payload=self.payload,
        )


@dataclass(slots=True)
class ContentPayload:
    filename: str
    content_type: str
    
    content: str

    def as_json(self) -> str:
        return Retort().dump(self)


@dataclass(slots=True)
class Content:
    text: str
    payload: Optional[ContentPayload] = None


@dataclass(slots=True)
class Embedder:
    client: openai.Client
    model: str

    async def get_embeddings_text(
        self,
        data: list[Content],
    ) -> list[Embedding]:
        result = self.client.embeddings.create(
            model=self.model,
            input=[obj.text for obj in data],
        )

        res = []

        for embedddings, content in zip(result.data, data, strict=True):
            res.append(
                Embedding(
                    vector=embedddings.embedding,
                    payload=content.payload.as_json() if content.payload else {},
                ),
            )

        return res
