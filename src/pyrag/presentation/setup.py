from fastapi import APIRouter, FastAPI

from pyrag.presentation.knowledge_base.crud import ROUTER as KBRouter
from pyrag.presentation.llm.request import ROUTER as LLMRequestRouter

def setup_routes(app: FastAPI):
    api_router = APIRouter(prefix = "/api")

    api_router.include_router(
        KBRouter,
        prefix = "/kb",
        tags=["KnowledgeBase"]
    )

    api_router.include_router(
        LLMRequestRouter,
        prefix = "/llm",
        tags=["LLM"]
    )


    app.include_router(api_router)