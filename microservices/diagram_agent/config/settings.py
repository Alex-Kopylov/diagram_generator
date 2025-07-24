"""Application settings and configuration."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-4o-mini", env="MODEL_NAME")
    
    # LangGraph Configuration
    graph_renderer: str = Field(default="png", env="GRAPH_RENDERER")
    langchain_tracing_v2: bool = Field(default=False, env="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    
    # Application Configuration
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Diagram Configuration
    default_output_format: str = Field(default="png", env="DEFAULT_OUTPUT_FORMAT")
    max_diagram_size: str = Field(default="10MB", env="MAX_DIAGRAM_SIZE")
    temp_dir: Path = Field(default=Path("/tmp/diagram_generator"), env="TEMP_DIR")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()