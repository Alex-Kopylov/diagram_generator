"""
LangGraph tools for diagram generation.

This module provides native LangGraph tools for direct graph construction.
"""

from typing import Any, Dict, List, Optional, Union
import os
import sys
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Import graph structures - create a local copy to avoid external dependencies
from core.graph_structure import Node, Edge, Cluster, Graph, Direction


class CreateNodeInput(BaseModel):
    """Input schema for creating a new graph node."""
    name: str = Field(..., description="Node name (e.g., 'diagrams.aws.compute.EC2')")
    id: Optional[str] = Field(None, description="Optional custom node ID")


class CreateEdgeInput(BaseModel):
    """Input schema for creating a new graph edge."""
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    forward: bool = Field(False, description="Enable forward direction arrow")
    reverse: bool = Field(False, description="Enable reverse direction arrow")


class CreateClusterInput(BaseModel):
    """Input schema for creating a cluster."""
    name: str = Field(..., description="Cluster name")
    node_ids: List[str] = Field(..., description="List of node IDs to include in cluster")


class BuildGraphInput(BaseModel):
    """Input schema for building a complete graph."""
    name: str = Field(..., description="Graph name")
    direction: Direction = Field(Direction.LEFT_RIGHT, description="Graph layout direction")
    nodes: List[Node] = Field(..., description="Array of node objects")
    edges: List[Union[Edge, Dict[str, Any]]] = Field(..., description="Array of edge objects or edge data")
    clusters: Optional[List[Union[Cluster, Dict[str, Any]]]] = Field(None, description="Array of cluster objects or cluster data")


class AddToGraphInput(BaseModel):
    """Input schema for adding components to existing graph."""
    graph: Graph = Field(..., description="Existing graph object")
    nodes: Optional[List[Node]] = Field(None, description="Nodes to add")
    edges: Optional[List[Union[Edge, Dict[str, Any]]]] = Field(None, description="Edges to add")
    clusters: Optional[List[Union[Cluster, Dict[str, Any]]]] = Field(None, description="Clusters to add")


class ValidationResult(BaseModel):
    """Graph validation result schema."""
    valid: bool = Field(..., description="Whether the graph is valid")
    errors: List[str] = Field(..., description="List of validation errors")


class ValidateGraphInput(BaseModel):
    """Input schema for graph validation."""
    graph_data: Dict[str, Any] = Field(..., description="Graph data as dictionary")


class GraphToJsonInput(BaseModel):
    """Input schema for graph to JSON conversion."""
    graph_data: Dict[str, Any] = Field(..., description="Graph data as dictionary")


class GenerateDiagramInput(BaseModel):
    """Input schema for diagram generation."""
    graph_data: Dict[str, Any] = Field(..., description="Graph data as dictionary")
    output_file: Optional[str] = Field(None, description="Optional output file name")


class DiagramResult(BaseModel):
    """Diagram generation result schema."""
    success: bool = Field(..., description="Whether diagram generation succeeded")
    file_path: Optional[str] = Field(None, description="Generated diagram file path")
    error: Optional[str] = Field(None, description="Error message if generation failed")


@tool(args_schema=CreateNodeInput) 
def create_node(input: CreateNodeInput) -> Node:
    """Create a new graph node.
    
    Args:
        input: Node creation parameters
        
    Returns:
        Node: Created node object
    """
    if input.id:
        return Node(name=input.name, id=input.id)
    else:
        return Node(name=input.name)


@tool(args_schema=CreateEdgeInput)
def create_edge(input: CreateEdgeInput) -> Dict[str, Any]:
    """Create a new graph edge between nodes.
    
    Args:
        input: Edge creation parameters
        
    Returns:
        Dict: Edge data with IDs and direction info
    """
    # Create dummy nodes for edge creation
    source_node = Node(name="temp", id=input.source_id)
    target_node = Node(name="temp", id=input.target_id)
    
    edge = Edge(
        source=source_node,
        target=target_node,
        forward=input.forward,
        reverse=input.reverse
    )
    
    return {
        "id": edge.id,
        "source_id": input.source_id,
        "target_id": input.target_id,
        "forward": input.forward,
        "reverse": input.reverse
    }


@tool(args_schema=CreateClusterInput)
def create_cluster(input: CreateClusterInput) -> Dict[str, Any]:
    """Create a cluster containing specified nodes.
    
    Args:
        input: Cluster creation parameters
        
    Returns:
        Dict: Cluster data with name, node IDs, and cluster ID
    """
    cluster = Cluster(name=input.name)
    return {
        "name": input.name,
        "node_ids": input.node_ids,
        "id": cluster.id
    }


@tool(args_schema=BuildGraphInput)
def build_graph(input: BuildGraphInput) -> Graph:
    """Build a complete graph from components.
    
    Args:
        input: Graph building parameters
        
    Returns:
        Graph: Complete graph object
    """
    # Create graph
    graph = Graph(name=input.name, direction=input.direction)
    
    # Create and add nodes
    node_map = {}
    for node in input.nodes:
        graph.add_node(node)
        node_map[node.id] = node
    
    # Create and add edges
    for edge_data in input.edges:
        if "source_id" in edge_data:
            # Handle edge data with node IDs
            source_node = node_map[edge_data["source_id"]]
            target_node = node_map[edge_data["target_id"]]
            edge = Edge(
                source=source_node,
                target=target_node,
                forward=edge_data.get("forward", False),
                reverse=edge_data.get("reverse", False)
            )
        else:
            # Handle full edge objects
            edge = Edge.model_validate(edge_data)
        
        graph.add_edge(edge)
    
    # Create and add clusters
    if input.clusters:
        for cluster_data in input.clusters:
            if "node_ids" in cluster_data:
                # Handle cluster data with node IDs
                cluster = Cluster(name=cluster_data["name"])
                for node_id in cluster_data["node_ids"]:
                    if node_id in node_map:
                        cluster.add_node(node_map[node_id])
            else:
                # Handle full cluster objects
                cluster = Cluster.model_validate(cluster_data)
            
            graph.add_cluster(cluster)
    
    return graph


@tool(args_schema=AddToGraphInput)
def add_to_graph(input: AddToGraphInput) -> Graph:
    """Add components to an existing graph.
    
    Args:
        input: Graph addition parameters
        
    Returns:
        Graph: Updated graph object
    """
    # Use the provided graph object
    graph = input.graph
    
    # Create node map for existing nodes
    node_map = {node.id: node for node in graph.nodes}
    
    # Add new nodes
    if input.nodes:
        for node in input.nodes:
            graph.add_node(node)
            node_map[node.id] = node
    
    # Add new edges
    if input.edges:
        for edge_data in input.edges:
            if "source_id" in edge_data:
                source_node = node_map[edge_data["source_id"]]
                target_node = node_map[edge_data["target_id"]]
                edge = Edge(
                    source=source_node,
                    target=target_node,
                    forward=edge_data.get("forward", False),
                    reverse=edge_data.get("reverse", False)
                )
            else:
                edge = Edge.model_validate(edge_data)
            
            graph.add_edge(edge)
    
    # Add new clusters
    if input.clusters:
        for cluster_data in input.clusters:
            if "node_ids" in cluster_data:
                cluster = Cluster(name=cluster_data["name"])
                for node_id in cluster_data["node_ids"]:
                    if node_id in node_map:
                        cluster.add_node(node_map[node_id])
            else:
                cluster = Cluster.model_validate(cluster_data)
            
            graph.add_cluster(cluster)
    
    return graph


@tool(args_schema=ValidateGraphInput)
def validate_graph(input: ValidateGraphInput) -> ValidationResult:
    """Validate graph structure and connections.
    
    Args:
        input: Graph validation parameters
        
    Returns:
        ValidationResult: Validation result with errors if any
    """
    graph = Graph.model_validate(input.graph_data)
    errors = []
    
    # Check for orphaned edges
    node_ids = {node.id for node in graph.nodes}
    for edge in graph.edges:
        if edge.source.id not in node_ids:
            errors.append(f"Edge references non-existent source node: {edge.source.id}")
        if edge.target.id not in node_ids:
            errors.append(f"Edge references non-existent target node: {edge.target.id}")
    
    # Check cluster nodes exist
    for cluster_name, cluster in graph.clusters.items():
        for node in cluster.nodes:
            if node.id not in node_ids:
                errors.append(f"Cluster '{cluster_name}' references non-existent node: {node.id}")
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)


@tool(args_schema=GraphToJsonInput)
def graph_to_json(input: GraphToJsonInput) -> str:
    """Convert graph to JSON string.
    
    Args:
        input: Graph to JSON conversion parameters
        
    Returns:
        str: JSON string representation of the graph
    """
    graph = Graph.model_validate(input.graph_data)
    return graph.model_dump_json(indent=2)


@tool(args_schema=GenerateDiagramInput)
def generate_diagram(input: GenerateDiagramInput) -> DiagramResult:
    """Generate diagram file from graph.
    
    Args:
        input: Diagram generation parameters
        
    Returns:
        DiagramResult: Generation result with success status and file path
    """
    try:
        graph = Graph.model_validate(input.graph_data)
        
        # Generate diagram using the existing to_diagrams method
        graph.to_diagrams()
        
        # Determine output file name
        if not input.output_file:
            output_file = f"{graph.name.lower().replace(' ', '_')}.png"
        else:
            output_file = input.output_file
        
        return DiagramResult(success=True, file_path=output_file, error=None)
    
    except Exception as e:
        return DiagramResult(success=False, file_path=None, error=str(e))


# List of all available tools for easy import
ALL_GRAPH_TOOLS = [
    create_node,
    create_edge,
    create_cluster,
    build_graph,
    add_to_graph,
    validate_graph,
    graph_to_json,
    generate_diagram
]