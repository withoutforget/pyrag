from dataclasses import dataclass
import os
import tomllib

from adaptix import Retort


@dataclass(slots=True)
class EmbeddingConfig:
    base_url: str
    api_key: str
    model: str

@dataclass(slots=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str



@dataclass(slots=True)
class QdrantConfig:
    host: str
    port: int
    dim: int

@dataclass(slots=True)
class Config:
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    llm: LLMConfig


def get_config() -> Config:
    config_path = os.getenv("CONFIG_FILE", "./infra/config.toml")
    with open(config_path, "rb") as f:
        raw_config = tomllib.load(f)
    return Retort().load(raw_config, Config)