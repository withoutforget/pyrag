from dataclasses import dataclass
import uuid
import openai
from qdrant_client.models import PointStruct


@dataclass(slots=True)
class Embedding:
    vector: list[float]
    payload: dict

    def toPointStruct(self) -> PointStruct:
        return PointStruct(
            id = uuid.uuid4(),
            vector = { "cvector": self.vector },
            payload = self.payload
        )

@dataclass(slots=True)
class Content:
    text: str
    payload: dict

@dataclass(slots=True)
class Embedder:
    client: openai.Client
    model: str

    async def get_embeddings_text(self, data: list[Content]) -> list[Embedding]:
        result = self.client.embeddings.create(
            model = self.model,
            input = [obj.text for obj in data]
        )

        res = []

        for embedddings, content in zip(result.data, data):
            res.append(Embedding(vector = embedddings.embedding, payload = content.payload))

        return res