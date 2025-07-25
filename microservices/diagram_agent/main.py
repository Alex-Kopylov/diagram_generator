"""
Main entry point for the LangGraph Diagram Agent service.

This service provides diagram generation capabilities using native LangGraph tools.
"""

import uvicorn

from config.logger import get_logger_settings, setup_logging
from config.settings import get_settings

# Initialize settings
settings = get_settings()
logger_settings = get_logger_settings()

# Setup logging
setup_logging(logger_settings)


def main():
    """Run the diagram agent service."""
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
