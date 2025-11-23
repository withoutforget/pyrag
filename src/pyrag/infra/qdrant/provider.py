from dishka import Provider, Scope
from dishka import provide_all

from pyrag.infra.qdrant.qdrant import Qdrant

class QdrantProdiver(Provider):
    qdrant_client = provide_all(Qdrant, scope = Scope.REQUEST)