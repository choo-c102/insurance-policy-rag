from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    voyage_api_key: str
    pdf_path: str
    persist_directory: str = "./chroma_langchain_db"
    collections_file: str = "collections.json"
    model_config = {"env_file": ".env", "extra": "ignore"}

settings = Settings()