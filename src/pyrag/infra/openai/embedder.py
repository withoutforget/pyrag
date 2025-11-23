from dataclasses import dataclass
import uuid
import openai
from qdrant_client.models import PointStruct


@dataclass(slots=True)
class Embedding:
    vector: list[float]
    value: str

    def toPointStruct(self) -> PointStruct:
        return PointStruct(
            id = uuid.uuid4(),
            vector = { "cvector": self.vector},
            payload = { "text": self.value }
        )

EmbedderClient = openai.Client

@dataclass(slots=True)
class Embedder:
    client: EmbedderClient
    model: str

    async def get_embeddings_text(self, data: list[str]) -> list[Embedding]:
        result = self.client.embeddings.create(
            model = self.model,
            input = data
        )

        res = []

        for embedddings, text in zip(result.data, data):
            res.append(Embedding(vector = embedddings, value = text))

        return res