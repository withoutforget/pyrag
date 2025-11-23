from dishka import Provider, Scope
from dishka import provide, provide_all
import openai

from pyrag.config import Config, EmbeddingConfig, LLMConfig
from pyrag.infra.openai.embedder import Embedder
from pyrag.infra.openai.llm import LLMClient, LLMRequest

import logging

class OpenAIProdiver(Provider):

    @provide(scope=Scope.REQUEST)
    async def get_embedder(self,       config: Config   ) -> Embedder:
        client = openai.Client(
            base_url = config.embedding.base_url,
            api_key = config.embedding.api_key,
        )
        return Embedder(
            client = client,
            model = config.embedding.model,
        )
    
    
    @provide(scope=Scope.REQUEST)
    async def get_llm(self,   config: Config) -> LLMRequest:
        client = LLMClient(
            base_url = config.llm.base_url,
            api_key = config.llm.api_key,
        )
        return LLMRequest(
            client = client,
            model = config.llm.model,
            prompt = config.llm.prompt
        )