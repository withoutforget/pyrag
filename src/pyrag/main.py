from fastapi import FastAPI
from dishka import make_async_container, Provider, provide, Scope
from dishka.integrations.fastapi import setup_dishka
import openai
from qdrant_client import QdrantClient
from pyrag.config import get_config, Config
from pyrag.presentation import router

class InfraProvider(Provider):
    @provide(scope=Scope.APP)
    async def get_qdrant(self, config: Config) -> QdrantClient:
        return QdrantClient(
            host = config.qdrant.host,
            port = config.qdrant.port,
        )
    @provide(scope=Scope.APP)
    async def get_openai(self, config: Config) -> openai.Client:
        return openai.Client(
            base_url = config.embedding.base_url,
            api_key = config.embedding.api_key,
        )
    
container = make_async_container(
    InfraProvider(),
    context = {
        Config: get_config(),
    }
)

app = FastAPI()

setup_dishka(container=container, app = app)

app.include_router(router=router)
