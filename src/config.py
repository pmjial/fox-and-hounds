from typing import Self

from pydantic import BaseModel, TypeAdapter


class ModelConfig(BaseModel):
    name: str
    base_url: str | None = None
    api_key: str
    temperature: float | None = None
    timeout: float = 120.0


class Config(BaseModel):
    @classmethod
    def load(cls, file: str = "config.json") -> Self:
        with open(file, "rb") as fp:
            return TypeAdapter(dict[str, ModelConfig]).validate_json(fp.read())
