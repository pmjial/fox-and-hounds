from pydantic import BaseModel, TypeAdapter


class ConnectionConfig(BaseModel):
    base_url: str | None = None
    api_key: str
    temperature: float | None = None
    timeout: float = 120.0


class ModelConfig(BaseModel):
    name: str
    conn: ConnectionConfig


class Config(BaseModel):
    @classmethod
    def load(cls, file: str = "config.json"):
        with open(file, "rb") as fp:
            return TypeAdapter(dict[str, ConnectionConfig]).validate_json(fp.read())
