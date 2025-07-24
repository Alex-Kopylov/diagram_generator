"""
Main entry point for the LangGraph Diagram Agent service.

This service provides diagram generation capabilities using native LangGraph tools
without requiring an external MCP server.
"""

import os
import uvicorn
from dotenv import load_dotenv

from api.endpoints import app

# Load environment variables
load_dotenv()


def main():
    """Run the diagram agent service."""
    port = int(os.getenv("PORT", "3502"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting LangGraph Diagram Agent on {host}:{port}")
    print("Features:")
    print("- Native LangGraph tools")
    print("- No MCP server dependency")
    print("- Direct graph construction")
    print("- Conversational diagram generation")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    main()