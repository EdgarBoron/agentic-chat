from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False)

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "llama-3.1-8b-instruct"
    searxng_url: str = "http://localhost:8080"
    chroma_persist_dir: str = "./data/chroma"
    checkpoint_db_path: str = "./data/checkpoints.db"
    reference_data_dir: str = "./data/reference"
    cors_origins: str = "http://localhost:3000"
    imagegen_url: str = "http://localhost:8000"
    images_db_path: str = "./data/images.db"
    images_dir: str = "./data/images"


settings = Settings()
