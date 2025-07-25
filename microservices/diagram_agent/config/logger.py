"""Logging configuration and setup using loguru."""

import sys
from pathlib import Path

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings


class LoggerSettings(BaseSettings):
    """Logger configuration settings."""

    log_level: str = Field(default="INFO")
    log_file: str | None = Field(default="logs/diagram_agent.log")
    debug: bool = Field(default=True)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


def setup_logging(settings: LoggerSettings | None = None) -> None:
    """
    Setup loguru logging configuration.
    
    Args:
        settings (Optional[LoggerSettings]): Logger settings. If None, uses default settings.
    """
    if settings is None:
        settings = LoggerSettings()

    logger.remove()

    log_level = "DEBUG" if settings.debug else settings.log_level

    logger.add(
        sys.stderr,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            settings.log_file,
            level=log_level,
            backtrace=True,
            diagnose=True,
        )

    logger.info(f"Logging configured with level: {log_level}")


def get_logger_settings() -> LoggerSettings:
    """Get logger settings instance."""
    return LoggerSettings()
