import logging
from fastapi import FastAPI
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI
import structlog

from pyrag.presentation.setup import setup_routes
from pyrag.main.di import container

def setup_app() -> FastAPI:
    app = FastAPI()

    setup_dishka(container=container, app=app)

    setup_routes(app)

    return app