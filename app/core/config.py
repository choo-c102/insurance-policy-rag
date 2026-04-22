from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    voyage_api_key: str
    pdf_path: str
    persist_directory: str = "./chroma_langchain_db"

    model_config = {"env_file": ".env", "extra": "ignore"}

settings = Settings()