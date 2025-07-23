"""Shared data models for the diagram agent."""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


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


class EdgeDirection(str, Enum):
    """Edge connection types."""
    RIGHT_TO_LEFT = ">>"
    LEFT_TO_RIGHT = "<<"
    UNDIRECTED = "-"


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
    provider: str = Field(..., description="Cloud provider (aws, gcp, azure, etc.)")
    resource_type: str = Field(..., description="Resource category (compute, database, etc.)")
    node_class: str = Field(..., description="Specific node class name")
    label: str = Field(..., description="Display label")
    cluster: Optional[str] = Field(default=None, description="Parent cluster name")


class ClusterSpec(BaseModel):
    """Specification for creating a diagram cluster."""
    name: str = Field(..., description="Cluster name/label")
    parent_cluster: Optional[str] = Field(default=None, description="Parent cluster for nesting")
    graph_attr: Optional[Dict[str, Any]] = Field(default=None, description="Custom attributes")


class EdgeSpec(BaseModel):
    """Specification for creating diagram edges."""
    left_node: str = Field(..., description="Left node identifier")
    right_node: str = Field(..., description="Right node identifier")
    direction: EdgeDirection = Field(default=EdgeDirection.UNDIRECTED, description="Connection type")
    label: Optional[str] = Field(default=None, description="Edge label")
    color: Optional[str] = Field(default=None, description="Edge color")
    style: EdgeStyle = Field(default=EdgeStyle.SOLID, description="Edge visual style")


class DiagramSpec(BaseModel):
    """Complete specification for a diagram."""
    name: str = Field(..., description="Diagram title")
    output_format: OutputFormat = Field(default=OutputFormat.PNG, description="Output format")
    filename: Optional[str] = Field(default=None, description="Custom filename")
    direction: Optional[Direction] = Field(default=None, description="Layout direction")
    graph_attr: Optional[Dict[str, Any]] = Field(default=None, description="Custom attributes")


class AgentState(BaseModel):
    """State model for LangGraph agent workflow."""
    ... # TODO: Implement