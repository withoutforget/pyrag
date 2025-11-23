from dishka import Provider, Scope
from dishka import provide, provide_all
import openai

from pyrag.config import Config, EmbeddingConfig, LLMConfig
from pyrag.infra.openai.embedder import Embedder, EmbedderClient
from pyrag.infra.openai.llm import LLMClient, LLMRequest

class OpenAIProdiver(Provider):

    @provide(scope=Scope.REQUEST)
    async def get_embedder_config(self, config: EmbeddingConfig) -> EmbedderClient:
        return EmbedderClient(
            base_url = config.base_url,
            api_key = config.api_key,
        )
    
    @provide(scope=Scope.REQUEST)
    async def get_embedder(self, 
                                  config: EmbeddingConfig,
                                  client: EmbedderClient) -> Embedder:
        return Embedder(
            client = client,
            model = config.model,
        )
    
    @provide(scope=Scope.REQUEST)
    async def get_llm_config(self, config: LLMConfig) -> LLMClient:
        return LLMClient(
            base_url = config.base_url,
            api_key = config.api_key,
        )
    
    @provide(scope=Scope.REQUEST)
    async def get_llm(self,
                       config: LLMConfig,
                       client: LLMClient) -> LLMRequest:
        raise NotImplementedError("Watch here.")
        return LLMRequest(
            client = client,
            model = config.model,
            promp = "..."
        )