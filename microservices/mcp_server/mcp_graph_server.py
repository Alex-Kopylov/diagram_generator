#!/usr/bin/env python3
"""
MCP Server for Graph Construction Tools

A stateless MCP server providing tools for constructing and manipulating
graph diagrams using the existing graph_structure.py classes.
"""

from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from .graph_structure import Node, Edge, Cluster, Graph, Direction


# Input/Output Schemas using Pydantic
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


class ValidateGraphInput(BaseModel):
    """Input schema for graph validation."""
    graph: Graph = Field(..., description="Graph object to validate")


class GraphToJsonInput(BaseModel):
    """Input schema for graph JSON conversion."""
    graph: Graph = Field(..., description="Graph object to serialize")


class GenerateDiagramInput(BaseModel):
    """Input schema for diagram generation."""
    graph: Graph = Field(..., description="Graph object to render")
    output_file: Optional[str] = Field(None, description="Optional output file name")


# Output Schemas
class EdgeData(BaseModel):
    """Edge data output schema."""
    id: str = Field(..., description="Edge unique identifier")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    forward: bool = Field(..., description="Forward direction enabled")
    reverse: bool = Field(..., description="Reverse direction enabled")


class ClusterData(BaseModel):
    """Cluster data output schema."""
    name: str = Field(..., description="Cluster name")
    node_ids: List[str] = Field(..., description="Node IDs in cluster")
    id: str = Field(..., description="Cluster unique identifier")


class ValidationResult(BaseModel):
    """Graph validation result schema."""
    valid: bool = Field(..., description="Whether the graph is valid")
    errors: List[str] = Field(..., description="List of validation errors")


class JsonResult(BaseModel):
    """JSON conversion result schema."""
    json_string: str = Field(..., description="JSON string representation of the graph")


class DiagramResult(BaseModel):
    """Diagram generation result schema."""
    success: bool = Field(..., description="Whether diagram generation succeeded")
    file_path: Optional[str] = Field(None, description="Generated diagram file path")
    error: Optional[str] = Field(None, description="Error message if generation failed")


class ErrorResult(BaseModel):
    """Error result schema."""
    error: str = Field(..., description="Error message describing what went wrong")


# Initialize the FastMCP server
mcp = FastMCP("graph-construction")


# Tool Handler Functions
async def handle_create_node(input_data: CreateNodeInput) -> Node:
    """Create a new graph node."""
    if input_data.id:
        return Node(name=input_data.name, id=input_data.id)
    return Node(name=input_data.name)


async def handle_create_edge(input_data: CreateEdgeInput) -> EdgeData:
    """Create a new graph edge."""
    # Create dummy nodes for edge creation
    source_node = Node(name="temp", id=input_data.source_id)
    target_node = Node(name="temp", id=input_data.target_id)
    
    edge = Edge(
        source=source_node,
        target=target_node,
        forward=input_data.forward,
        reverse=input_data.reverse
    )
    
    return EdgeData(
        id=edge.id,
        source_id=input_data.source_id,
        target_id=input_data.target_id,
        forward=input_data.forward,
        reverse=input_data.reverse
    )


async def handle_create_cluster(input_data: CreateClusterInput) -> ClusterData:
    """Create a cluster."""
    cluster = Cluster(name=input_data.name)
    return ClusterData(
        name=input_data.name,
        node_ids=input_data.node_ids,
        id=cluster.id
    )


async def handle_build_graph(input_data: BuildGraphInput) -> Graph:
    """Build complete graph from components."""
    # Create graph
    graph = Graph(name=input_data.name, direction=input_data.direction)
    
    # Create and add nodes
    node_map = {}
    for node in input_data.nodes:
        graph.add_node(node)
        node_map[node.id] = node
    
    # Create and add edges
    for edge_data in input_data.edges:
        if isinstance(edge_data, dict) and "source_id" in edge_data:
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
            edge = edge_data if isinstance(edge_data, Edge) else Edge.model_validate(edge_data)
        
        graph.add_edge(edge)
    
    # Create and add clusters
    if input_data.clusters:
        for cluster_data in input_data.clusters:
            if isinstance(cluster_data, dict) and "node_ids" in cluster_data:
                # Handle cluster data with node IDs
                cluster = Cluster(name=cluster_data["name"])
                for node_id in cluster_data["node_ids"]:
                    if node_id in node_map:
                        cluster.add_node(node_map[node_id])
            else:
                # Handle full cluster objects
                cluster = cluster_data if isinstance(cluster_data, Cluster) else Cluster.model_validate(cluster_data)
            
            graph.add_cluster(cluster)
    
    return graph


async def handle_add_to_graph(input_data: AddToGraphInput) -> Graph:
    """Add components to existing graph."""
    graph = input_data.graph
    
    # Create node map for existing nodes
    node_map = {node.id: node for node in graph.nodes}
    
    # Add new nodes
    if input_data.nodes:
        for node in input_data.nodes:
            graph.add_node(node)
            node_map[node.id] = node
    
    # Add new edges
    if input_data.edges:
        for edge_data in input_data.edges:
            if isinstance(edge_data, dict) and "source_id" in edge_data:
                source_node = node_map[edge_data["source_id"]]
                target_node = node_map[edge_data["target_id"]]
                edge = Edge(
                    source=source_node,
                    target=target_node,
                    forward=edge_data.get("forward", False),
                    reverse=edge_data.get("reverse", False)
                )
            else:
                edge = edge_data if isinstance(edge_data, Edge) else Edge.model_validate(edge_data)
            
            graph.add_edge(edge)
    
    # Add new clusters
    if input_data.clusters:
        for cluster_data in input_data.clusters:
            if isinstance(cluster_data, dict) and "node_ids" in cluster_data:
                cluster = Cluster(name=cluster_data["name"])
                for node_id in cluster_data["node_ids"]:
                    if node_id in node_map:
                        cluster.add_node(node_map[node_id])
            else:
                cluster = cluster_data if isinstance(cluster_data, Cluster) else Cluster.model_validate(cluster_data)
            
            graph.add_cluster(cluster)
    
    return graph


async def handle_validate_graph(input_data: ValidateGraphInput) -> ValidationResult:
    """Validate graph structure."""
    graph = input_data.graph
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


async def handle_graph_to_json(input_data: GraphToJsonInput) -> JsonResult:
    """Convert graph to JSON."""
    return JsonResult(json_string=input_data.graph.model_dump_json(indent=2))


async def handle_generate_diagram(input_data: GenerateDiagramInput) -> DiagramResult:
    """Generate diagram from graph."""
    try:
        # Generate diagram using the existing to_diagrams method
        input_data.graph.to_diagrams()
        
        # Determine output file name
        output_file = input_data.output_file
        if not output_file:
            output_file = f"{input_data.graph.name.lower().replace(' ', '_')}.png"
        
        return DiagramResult(success=True, file_path=output_file, error=None)
    
    except Exception as e:
        return DiagramResult(success=False, file_path=None, error=str(e))


# Tool registrations using FastMCP decorators
@mcp.tool()
async def create_node(name: str, id: Optional[str] = None) -> Node:
    """Create a new graph node."""
    input_data = CreateNodeInput(name=name, id=id)
    return await handle_create_node(input_data)


@mcp.tool()
async def create_edge(source_id: str, target_id: str, forward: bool = False, reverse: bool = False) -> EdgeData:
    """Create a new graph edge between nodes."""
    input_data = CreateEdgeInput(source_id=source_id, target_id=target_id, forward=forward, reverse=reverse)
    return await handle_create_edge(input_data)


@mcp.tool()
async def create_cluster(name: str, node_ids: List[str]) -> ClusterData:
    """Create a cluster containing specified nodes."""
    input_data = CreateClusterInput(name=name, node_ids=node_ids)
    return await handle_create_cluster(input_data)


@mcp.tool()
async def build_graph(name: str, direction: Direction, nodes: List[Node], edges: List[Union[Edge, Dict[str, Any]]], clusters: Optional[List[Union[Cluster, Dict[str, Any]]]] = None) -> Graph:
    """Build a complete graph from components."""
    input_data = BuildGraphInput(name=name, direction=direction, nodes=nodes, edges=edges, clusters=clusters)
    return await handle_build_graph(input_data)


@mcp.tool()
async def add_to_graph(graph: Graph, nodes: Optional[List[Node]] = None, edges: Optional[List[Union[Edge, Dict[str, Any]]]] = None, clusters: Optional[List[Union[Cluster, Dict[str, Any]]]] = None) -> Graph:
    """Add components to an existing graph."""
    input_data = AddToGraphInput(graph=graph, nodes=nodes, edges=edges, clusters=clusters)
    return await handle_add_to_graph(input_data)


@mcp.tool()
async def validate_graph(graph: Graph) -> ValidationResult:
    """Validate graph structure and connections."""
    input_data = ValidateGraphInput(graph=graph)
    return await handle_validate_graph(input_data)


@mcp.tool()
async def graph_to_json(graph: Graph) -> JsonResult:
    """Convert graph to JSON string."""
    input_data = GraphToJsonInput(graph=graph)
    return await handle_graph_to_json(input_data)


@mcp.tool()
async def generate_diagram(graph: Graph, output_file: Optional[str] = None) -> DiagramResult:
    """Generate diagram file from graph."""
    input_data = GenerateDiagramInput(graph=graph, output_file=output_file)
    return await handle_generate_diagram(input_data)


if __name__ == "__main__":
    mcp.run()