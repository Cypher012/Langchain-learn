from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    OPENAI_API_KEY: str
    TAVILY_API_KEY: str
    LANGSMITH_API_KEY: str
    LANGSMITH_TRACING: bool
    LANGSMITH_PROJECT: str
    LANGSMITH_ENDPOINT: str


settings = Settings()
