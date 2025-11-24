from dishka import AsyncContainer, make_async_container

from pyrag.config import Config, get_config
from pyrag.infra.openai.provider import OpenAIProdiver
from pyrag.infra.parsers.provider import ParserProdiver
from pyrag.infra.qdrant.provider import QdrantProdiver

def get_container(config: Config) -> AsyncContainer:
    return make_async_container(
        OpenAIProdiver(),
        ParserProdiver(),
        QdrantProdiver(),
        context={
            Config: config,
        },
    )

container: AsyncContainer = get_container(get_config())