"""Application settings and configuration."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # OpenAI Configuration
    model_api_key: SecretStr | None = Field(default="test")
    fallback_model_api_key: SecretStr | None = Field(default="test")
    model_name: str = Field(default="gpt-4.1", env="MODEL_NAME")
    fallback_model_name: str = Field(default="claude-3-7-sonnet")
    temperature: float = Field(default=0.0, env="TEMPERATURE")

    # LangGraph Configuration
    graph_renderer: str = Field(default="png", env="GRAPH_RENDERER")
    langchain_tracing_v2: bool = Field(default=False, env="LANGCHAIN_TRACING_V2")
   
    # Application Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=3502, env="PORT")
    reload: bool = Field(default=True, env="RELOAD")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")  # Updated to match logger format
    debug: bool = Field(default=False, env="DEBUG")

    # Diagram Configuration
    default_output_format: str = Field(default="png", env="DEFAULT_OUTPUT_FORMAT")
    temp_dir: Path = Field(default=Path("/tmp/diagram_generator"), env="TEMP_DIR")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()
