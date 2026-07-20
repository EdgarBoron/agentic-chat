from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False)

    # Chat/agent LLM. "vllm" (default) uses the local self-hosted vLLM
    # server below; "openai" calls the real OpenAI API with openai_api_key;
    # "anthropic" calls the real Anthropic API with anthropic_api_key.
    llm_provider: str = "vllm"
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "llama-3.1-8b-instruct"
    openai_api_key: str = ""
    openai_model_name: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model_name: str = "claude-sonnet-5"
    searxng_url: str = "http://localhost:8080"
    chroma_persist_dir: str = "./data/chroma"
    checkpoint_db_path: str = "./data/checkpoints.db"
    reference_data_dir: str = "./data/reference"
    artist_styles_file: str = "./data/reference/artist-photographer-styles.md"
    cors_origins: str = "http://localhost:3000"
    imagegen_url: str = "http://localhost:8000"
    images_db_path: str = "./data/images.db"
    images_dir: str = "./data/images"

    @property
    def uses_local_vllm(self) -> bool:
        """False when the chat LLM is a remote provider (e.g. OpenAI) —
        there's then no local GPU chat model to pause around image
        generation."""
        return self.llm_provider == "vllm"


settings = Settings()
