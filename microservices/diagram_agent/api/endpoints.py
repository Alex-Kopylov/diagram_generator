"""
FastAPI endpoints for the LangGraph Diagram Agent.

Provides HTTP endpoints for diagram generation and health checks using native tools.
"""

import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from loguru import logger

from agents.diagram_agent import DiagramGenerationResult, create_diagram_agent

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph Diagram Agent",
    description="Intelligent diagram generation using LangGraph with native tools",
    version="2.0.0"
)

# Initialize the diagram agent
diagram_agent = create_diagram_agent()


# Request/Response Models
class DiagramRequest(BaseModel):
    """Request schema for diagram generation."""
    message: str = Field(..., description="User message describing the diagram to create")




class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation")


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    message: str = Field(..., description="Agent response")
    session_id: Optional[str] = Field(None, description="Session ID")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    return HealthResponse(
        status="healthy",
    )


# Diagram generation endpoint (stateless)
@app.post("/generate-diagram", response_model=DiagramGenerationResult)
async def generate_diagram(request: DiagramRequest):
    """
    Generate a diagram from a user message.
    
    This endpoint provides stateless diagram generation using the LangGraph agent
    """
    logger.info(f"Diagram generation request received: {request.message}")
    try:
        # Use the native diagram agent
        result = await diagram_agent.generate_diagram(
            message=request.message
        )
        logger.info(f"Diagram generation request successful: {result.success}")
        return result
    
    except Exception as e:
        logger.exception(f"Diagram generation request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Diagram generation failed: {str(e)}")


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint for conversational diagram generation.
    
    This endpoint provides conversational interactions with the diagram agent
    using native tools.
    """
    logger.info(f"Chat request received for session {request.session_id}: {request.message}")
    try:
        raise NotImplementedError("Chat endpoint is not implemented")
    except Exception as e:
        logger.exception(f"Chat request failed for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "LangGraph Diagram Agent",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate_diagram": "/generate-diagram",
            "chat": "/chat",
            "docs": "/docs"
        },
        "features": [
            "Conversational diagram generation",
            "Direct graph construction"
        ]
    }