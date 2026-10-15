from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    TAVILY_API_KEY: str
    PINECONE_INDEX_NAME: str = "medium-blogs-embeddings-index"


settings = Settings()
