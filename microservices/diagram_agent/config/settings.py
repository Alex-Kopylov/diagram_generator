"""Application settings and configuration."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # OpenAI Configuration
    openai_api_key: Optional[SecretStr] = Field(default="test", env="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-4.1-nano-2025-04-14", env="MODEL_NAME")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    
    # LangGraph Configuration
    graph_renderer: str = Field(default="png", env="GRAPH_RENDERER")
    langchain_tracing_v2: bool = Field(default=False, env="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    
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