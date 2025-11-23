from dishka import Provider, Scope, provide
from dishka import provide_all
from qdrant_client import QdrantClient

from pyrag.config import Config, QdrantConfig
from pyrag.infra.qdrant.qdrant import Qdrant

class QdrantProdiver(Provider):
    qdrant_client = provide_all(Qdrant, scope = Scope.REQUEST)

    @provide(scope=Scope.APP)
    async def get_qdrant_cfg(self, config: Config) -> QdrantConfig:
        return config.qdrant

    @provide(scope=Scope.REQUEST)
    async def get_qdrant(self, config: Config) -> QdrantClient:
        return QdrantClient(
            host = config.qdrant.host,
            port = config.qdrant.port,
        )