import structlog
from pyrag.main.web import setup_app
from pyrag.infra.logging import setup_logger
from pyrag.config import config

setup_logger(config.logger)
logger = structlog.get_logger(__name__)
logger.info("logger set up")
app = setup_app()
