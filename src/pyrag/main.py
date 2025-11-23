from fastapi import FastAPI
from dishka import make_async_container
from dishka.integrations.fastapi import setup_dishka
from pyrag.config import get_config, Config
from pyrag.infra.openai.provider import OpenAIProdiver
from pyrag.infra.parsers.provider import ParserProdiver
from pyrag.infra.qdrant.provider import QdrantProdiver
from pyrag.presentation import router

    
container = make_async_container(
    OpenAIProdiver(),
    ParserProdiver(),
    QdrantProdiver(),
    context = {
        Config: get_config(),
    }
)

app = FastAPI()

setup_dishka(container=container, app = app)

app.include_router(router=router)
