import logging

import structlog

from pyrag.config import LoggerConfig


def setup_logger(config: LoggerConfig):
    # Shared processors for all environments
    common_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Configure standard logging
    handler = logging.StreamHandler()
    handler.setLevel(level=config.level)

    if config.debug:
        # Development: Pretty-printed coloured logs
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=common_processors,
        )
    else:
        # Production: JSON output
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=common_processors,
        )

    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    structlog.configure(
        processors=[
            *common_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

        
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = root_logger.handlers
    uvicorn_logger.setLevel(root_logger.level)
    uvicorn_logger.propagate = True  

    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers = root_logger.handlers
    uvicorn_access.setLevel(root_logger.level)
    uvicorn_access.propagate = True