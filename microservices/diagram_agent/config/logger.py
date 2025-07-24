"""Logging configuration and setup using loguru."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings


class LoggerSettings(BaseSettings):
    """Logger configuration settings."""
    
    # Logging Configuration
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default="logs/diagram_agent.log")
    
    # LLM Logging Configuration
    log_llm_calls: bool = Field(default=True)
    log_llm_responses: bool = Field(default=True)
    log_llm_timing: bool = Field(default=True)
    
    # Debug Configuration
    debug: bool = Field(default=False)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


def setup_logging(settings: Optional[LoggerSettings] = None) -> None:
    """
    Setup loguru logging configuration.
    
    Args:
        settings (Optional[LoggerSettings]): Logger settings. If None, uses default settings.
    """
    if settings is None:
        settings = LoggerSettings()
    
    # Remove default handler
    logger.remove()
    
    # Set log level based on debug mode
    log_level = "DEBUG" if settings.debug else settings.log_level
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # Add file handler if log_file is specified
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


def log_llm_call(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    **kwargs
) -> None:
    """
    Log LLM call details.
    
    Args:
        model (str): The model being called.
        prompt (str): The prompt being sent.
        temperature (float): Temperature setting.
        max_tokens (Optional[int]): Maximum tokens setting.
        **kwargs: Additional parameters.
    """
    settings = get_logger_settings()
    
    if not settings.log_llm_calls:
        return
    
    log_data = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs
    }
    
    logger.info(f"LLM Call: {log_data}")
    
    if settings.debug:
        logger.debug(f"LLM Prompt: {prompt}")


def log_llm_response(
    model: str,
    response: str,
    tokens_used: Optional[int] = None,
    duration_ms: Optional[float] = None,
    **kwargs
) -> None:
    """
    Log LLM response details.
    
    Args:
        model (str): The model that responded.
        response (str): The response content.
        tokens_used (Optional[int]): Number of tokens used.
        duration_ms (Optional[float]): Response duration in milliseconds.
        **kwargs: Additional response metadata.
    """
    settings = get_logger_settings()
    
    if not settings.log_llm_responses:
        return
    
    log_data = {
        "model": model,
        **kwargs
    }
    
    if duration_ms is not None and settings.log_llm_timing:
        log_data["duration_ms"] = duration_ms
    
    logger.info(f"LLM Response: {log_data}")
    
    if settings.debug:
        logger.debug(f"LLM Response Content: {response}")


def log_llm_error(
    model: str,
    error: Exception,
    prompt: Optional[str] = None,
    **kwargs
) -> None:
    """
    Log LLM error details.
    
    Args:
        model (str): The model that failed.
        error (Exception): The error that occurred.
        prompt (Optional[str]): The prompt that caused the error.
        **kwargs: Additional error context.
    """
    log_data = {
        "model": model,
        "error_type": type(error).__name__,
        "error_message": str(error),
        **kwargs
    }
    
    logger.error(f"LLM Error: {log_data}")
    
    if prompt:
        logger.error(f"Failed LLM Prompt: {prompt}")