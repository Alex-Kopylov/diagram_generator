"""
Diagram generation tools.

This package provides native LangGraph tools for diagram generation,
eliminating the need for MCP server communication.
"""

from .graph_tools import (
    create_node,
    create_edge,
    create_cluster,
    build_graph,
    add_to_graph,
    validate_graph,
    graph_to_json,
    generate_diagram,
    ALL_GRAPH_TOOLS
)

# Keep MCP client for backward compatibility if needed
from .mcp_client import MCPNetworkClient, setup_mcp_client

__all__ = [
    "create_node",
    "create_edge", 
    "create_cluster",
    "build_graph",
    "add_to_graph",
    "validate_graph",
    "graph_to_json",
    "generate_diagram",
    "ALL_GRAPH_TOOLS",
    # Legacy MCP client
    "MCPNetworkClient", 
    "setup_mcp_client"
]