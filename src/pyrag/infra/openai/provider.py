from dishka import Provider, Scope
from dishka import provide, provide_all
import openai

from pyrag.config import Config, EmbeddingConfig, LLMConfig
from pyrag.infra.openai.embedder import Embedder, EmbedderClient
from pyrag.infra.openai.llm import LLMClient, LLMRequest

class OpenAIProdiver(Provider):

    @provide(scope=Scope.REQUEST)
    async def get_embedder_config(self, config: Config) -> EmbedderClient:
        return EmbedderClient(
            base_url = config.embedding.base_url,
            api_key = config.embedding.api_key,
        )
    
    @provide(scope=Scope.REQUEST)
    async def get_embedder(self, 
                                  config: Config,
                                  client: EmbedderClient) -> Embedder:
        return Embedder(
            client = client,
            model = config.embedding.model,
        )
    
    @provide(scope=Scope.REQUEST)
    async def get_llm_config(self, config: Config) -> LLMClient:
        return LLMClient(
            base_url = config.llm.base_url,
            api_key = config.llm.api_key,
        )
    
    @provide(scope=Scope.REQUEST)
    async def get_llm(self,
                       config: Config,
                       client: LLMClient) -> LLMRequest:
        return LLMRequest(
            client = client,
            model = config.model,
            promp = "..."
        )