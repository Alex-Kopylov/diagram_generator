"""
FastAPI endpoints for the LangGraph Diagram Agent.

Provides HTTP endpoints for diagram generation and health checks using native tools.
"""

import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..agents.diagram_agent import create_diagram_agent

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
    output_file: Optional[str] = Field(None, description="Optional output file name")


class DiagramResponse(BaseModel):
    """Response schema for diagram generation."""
    success: bool = Field(..., description="Whether the diagram was generated successfully")
    message: str = Field(..., description="Success or error message")
    file_path: Optional[str] = Field(None, description="Path to generated diagram file")
    graph_data: Optional[Dict[str, Any]] = Field(None, description="Graph data used for diagram")


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
    tools_available: int = Field(..., description="Number of available native tools")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from ..tools.graph_tools import ALL_GRAPH_TOOLS
    return HealthResponse(
        status="healthy",
        tools_available=len(ALL_GRAPH_TOOLS)
    )


# Diagram generation endpoint (stateless)
@app.post("/generate-diagram", response_model=DiagramResponse)
async def generate_diagram(request: DiagramRequest):
    """
    Generate a diagram from a user message.
    
    This endpoint provides stateless diagram generation using the LangGraph agent
    """
    try:
        # Use the native diagram agent
        result = await diagram_agent.generate_diagram(
            message=request.message,
            output_file=request.output_file
        )
        
        return DiagramResponse(
            success=result.success,
            message=result.message,
            file_path=result.file_path,
            graph_data=result.graph_data
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagram generation failed: {str(e)}")


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint for conversational diagram generation.
    
    This endpoint provides conversational interactions with the diagram agent
    using native tools.
    """
    try:
        # Use the native diagram agent for chat
        response = await diagram_agent.chat(
            message=request.message,
            session_id=request.session_id
        )
        
        return ChatResponse(
            message=response,
            session_id=request.session_id
        )
    
    except Exception as e:
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