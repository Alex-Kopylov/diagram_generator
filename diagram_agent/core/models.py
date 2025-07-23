"""Shared data models for the diagram agent."""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import importlib


class OutputFormat(str, Enum):
    """Supported output formats for diagrams."""
    PNG = "png"
    SVG = "svg"
    JPG = "jpg"
    PDF = "pdf"
    DOT = "dot"


class Direction(str, Enum):
    """Diagram layout directions."""
    TOP_BOTTOM = "TB"
    BOTTOM_TOP = "BT"
    LEFT_RIGHT = "LR"
    RIGHT_LEFT = "RL"


class ConnectionType(str, Enum):
    """Connection types for programmatic diagram construction."""
    CONNECT = "connect"
    FORWARD = "forward"
    REVERSE = "reverse"
    BIDIRECTIONAL = "both"


class EdgeStyle(str, Enum):
    """Edge visual styles."""
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    BOLD = "bold"


# Request Models
class DiagramRequest(BaseModel):
    """Request model for diagram generation."""
    description: str = Field(..., description="Natural language description of the diagram")
    output_format: OutputFormat = Field(default=OutputFormat.PNG, description="Output format")
    direction: Optional[Direction] = Field(default=None, description="Layout direction")


class AssistantRequest(BaseModel):
    """Request model for assistant endpoint."""
    message: str = Field(..., description="User message or query")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Conversation context")


# Response Models
class DiagramResponse(BaseModel):
    """Response model for successful diagram generation."""
    success: bool = True
    execution_plan: List[str] = Field(..., description="Steps executed by the agent")
    description: str = Field(..., description="Generated diagram description")
    format: OutputFormat = Field(..., description="Output format used")


class AssistantResponse(BaseModel):
    """Response model for assistant endpoint."""
    message: str = Field(..., description="Assistant response")
    diagram_generated: bool = Field(default=False, description="Whether a diagram was created")
    execution_plan: Optional[List[str]] = Field(default=None, description="Execution steps if diagram was generated")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


# Internal Models
class NodeSpec(BaseModel):
    """Specification for creating a diagram node."""
    diagrams_path: str = Field(..., description="Pythonic path like 'diagrams.aws.compute.EC2'")
    label: str = Field(..., description="Display label")
    node_id: str = Field(..., description="Unique identifier for connections")
    cluster: Optional[str] = Field(default=None, description="Parent cluster name")
    
    @field_validator('diagrams_path')
    @classmethod
    def validate_diagrams_path(cls, v: str) -> str:
        """Validate that the diagrams path exists in the library."""
        try:
            parts = v.split('.')
            if len(parts) < 4 or parts[0] != 'diagrams':
                raise ValueError("Path must start with 'diagrams.' and have at least 4 parts")
            
            module_path = '.'.join(parts[:-1])
            class_name = parts[-1]
            module = importlib.import_module(module_path)
            
            if not hasattr(module, class_name):
                raise ValueError(f"Class {class_name} not found in {module_path}")
            return v
        except ImportError:
            raise ValueError(f"Invalid diagrams path: {v}")


class ClusterSpec(BaseModel):
    """Specification for creating a diagram cluster."""
    name: str = Field(..., description="Cluster name/label")
    parent_cluster: Optional[str] = Field(default=None, description="Parent cluster for nesting")


class EdgeSpec(BaseModel):
    """Specification for creating diagram connections."""
    source_node_id: str = Field(..., description="Source node identifier")
    target_node_id: str = Field(..., description="Target node identifier")
    connection_type: ConnectionType = Field(default=ConnectionType.CONNECT, description="Connection type")
    label: Optional[str] = Field(default=None, description="Edge label")
    color: Optional[str] = Field(default=None, description="Edge color")
    style: EdgeStyle = Field(default=EdgeStyle.SOLID, description="Edge visual style")


class DiagramSpec(BaseModel):
    """Minimal specification for diagram creation."""
    name: str = Field(..., description="Diagram title")
    direction: Optional[Direction] = Field(default=None, description="Layout direction")


class AgentState(BaseModel):
    """State model for LangGraph agent workflow."""
    ... # TODO: Implement