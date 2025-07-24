"""
LangGraph tools for diagram generation.

This module provides native LangGraph tools for direct graph construction.
"""

from typing import Any, Dict, List, Optional, Union
import os
import sys
import pkgutil
import importlib
import inspect
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from loguru import logger

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
    graph: Graph = Field(..., description="Graph")
    output_file: Optional[str] = Field(None, description="Optional output file name")


class DiagramResult(BaseModel):
    """Diagram generation result schema."""
    success: bool = Field(..., description="Whether diagram generation succeeded")
    file_path: str | None = Field(None, description="Generated diagram file path")
    bytestring: bytes | None = Field(None, description="Generated diagram bytestring")
    error: str | None = Field(None, description="Error message if generation failed")


class ListResourcesByProviderInput(BaseModel):
    """Input schema for listing resources by provider."""
    provider: str = Field(..., description="Provider name (e.g., 'aws', 'gcp', 'azure')")


class ListNodesByResourceInput(BaseModel):
    """Input schema for listing nodes by provider and resource."""
    provider: str = Field(..., description="Provider name (e.g., 'aws', 'gcp', 'azure')")
    resource: str = Field(..., description="Resource category (e.g., 'compute', 'database', 'network')")


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
def create_edge(input: CreateEdgeInput) -> Edge:
    """Create a new graph edge between nodes.
    
    Args:
        input: Edge creation parameters
        
    Returns:
        Edge: Created edge object
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
    
    return edge


@tool(args_schema=CreateClusterInput)
def create_cluster(input: CreateClusterInput) -> Cluster:
    """Create a cluster containing specified nodes.
    
    Args:
        input: Cluster creation parameters
        
    Returns:
        Cluster: Created cluster object
    """
    cluster = Cluster(name=input.name)
    # Note: node_ids are provided but nodes need to be added separately
    # since we don't have access to the actual Node objects here
    return cluster


@tool(args_schema=BuildGraphInput)
def build_graph(input: BuildGraphInput) -> Graph:
    """Build a complete graph from components.
    
    Args:
        input: Graph building parameters
        
    Returns:
        Graph: Complete graph object
    """
    logger.debug(f"Building graph: name={input.name}, nodes={len(input.nodes)}, edges={len(input.edges)}")
    
    # Create graph
    graph = Graph(name=input.name, direction=input.direction)
    
    # Create and add nodes
    node_map = {}
    for node in input.nodes:
        graph.add_node(node)
        node_map[node.id] = node
        logger.debug(f"Added node to graph: {node.id}")
    
    # Create and add edges
    for edge_data in input.edges:
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
        elif isinstance(edge_data, Edge):
            # Handle Edge objects directly
            edge = edge_data
        else:
            # Handle dict objects as full edge data
            edge = Edge.model_validate(edge_data)
        
        graph.add_edge(edge)
    
    # Create and add clusters
    if input.clusters:
        for cluster_data in input.clusters:
            if isinstance(cluster_data, dict) and "node_ids" in cluster_data:
                # Handle cluster data with node IDs
                cluster = Cluster(name=cluster_data["name"])
                for node_id in cluster_data["node_ids"]:
                    if node_id in node_map:
                        cluster.add_node(node_map[node_id])
            elif isinstance(cluster_data, Cluster):
                # Handle Cluster objects directly
                cluster = cluster_data
            else:
                # Handle dict objects as full cluster data
                cluster = Cluster.model_validate(cluster_data)
            
            graph.add_cluster(cluster)
    
    logger.info(f"Graph built successfully: {graph.name} with {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return graph


@tool(args_schema=AddToGraphInput)
def add_to_graph(input: AddToGraphInput) -> Graph:
    """Add components to an existing graph.
    
    Args:
        input: Graph addition parameters
        
    Returns:
        Graph: Updated graph object
    """
    logger.debug(f"Adding to graph: nodes={len(input.nodes or [])}, edges={len(input.edges or [])}")
    
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
            if isinstance(edge_data, dict) and "source_id" in edge_data:
                source_node = node_map[edge_data["source_id"]]
                target_node = node_map[edge_data["target_id"]]
                edge = Edge(
                    source=source_node,
                    target=target_node,
                    forward=edge_data.get("forward", False),
                    reverse=edge_data.get("reverse", False)
                )
            elif isinstance(edge_data, Edge):
                # Handle Edge objects directly
                edge = edge_data
            else:
                edge = Edge.model_validate(edge_data)
            
            graph.add_edge(edge)
    
    # Add new clusters
    if input.clusters:
        for cluster_data in input.clusters:
            if isinstance(cluster_data, dict) and "node_ids" in cluster_data:
                cluster = Cluster(name=cluster_data["name"])
                for node_id in cluster_data["node_ids"]:
                    if node_id in node_map:
                        cluster.add_node(node_map[node_id])
            elif isinstance(cluster_data, Cluster):
                # Handle Cluster objects directly
                cluster = cluster_data
            else:
                cluster = Cluster.model_validate(cluster_data)
            
            graph.add_cluster(cluster)
    
    logger.info(f"Components added to graph successfully: {graph.name}")
    return graph


@tool(args_schema=ValidateGraphInput)
def validate_graph(input: ValidateGraphInput) -> ValidationResult:
    """Validate graph structure and connections.
    
    Args:
        input: Graph validation parameters
        
    Returns:
        ValidationResult: Validation result with errors if any
    """
    logger.debug("Validating graph structure")
    
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
    
    result = ValidationResult(valid=len(errors) == 0, errors=errors)
    if result.valid:
        logger.info("Graph validation successful")
    else:
        logger.warning(f"Graph validation failed with {len(errors)} errors: {errors}")
    
    return result




@tool(args_schema=GenerateDiagramInput)
def generate_diagram(input: GenerateDiagramInput) -> DiagramResult:
    """Generate diagram file from graph.
    
    Args:
        input: Diagram generation parameters
        
    Returns:
        DiagramResult: Generation result with success status and file path
    """
    try:
        logger.debug(f"Generating diagram, output_file={input.output_file}")
        
        # Generate diagram using the existing to_diagrams method
        diagram = input.graph.to_diagrams()
        
        # Determine output file name
        if not input.output_file:
            output_file = f"{input.graph.name.lower().replace(' ', '_')}.png"
        else:
            output_file = input.output_file
        
        logger.info(f"Diagram generated successfully: {output_file}")
        return DiagramResult(success=True, file_path=output_file, error=None,
                             bytestring=diagram._repr_png_())
    
    except Exception as e:
        logger.exception(f"Diagram generation failed: {str(e)}")
        return DiagramResult(success=False, file_path=None, error=str(e))


@tool
def list_all_providers() -> List[str]:
    """List all available cloud providers in the diagrams package.
    
    Returns:
        List[str]: List of available providers (e.g., ['aws', 'gcp', 'azure', 'alibaba'])
    """
    try:
        import diagrams
        providers = []
        
        # Get the diagrams package path
        diagrams_path = diagrams.__path__
        
        # Walk through all modules in diagrams package
        for importer, modname, ispkg in pkgutil.iter_modules(diagrams_path):
            # Skip __init__ and other special modules
            if not modname.startswith('_') and ispkg:
                # Check if it's a provider by looking for submodules
                try:
                    provider_module = importlib.import_module(f'diagrams.{modname}')
                    if hasattr(provider_module, '__path__'):
                        providers.append(modname)
                except ImportError:
                    continue
        
        providers.sort()
        logger.info(f"Found {len(providers)} providers: {providers}")
        return providers
        
    except Exception as e:
        logger.exception(f"Failed to list providers: {str(e)}")
        return []


@tool(args_schema=ListResourcesByProviderInput)
def list_resources_by_provider(input: ListResourcesByProviderInput) -> List[str]:
    """List all resource categories for a specific provider.
    
    Args:
        input: Provider specification
        
    Returns:
        List[str]: List of resource categories (e.g., ['compute', 'database', 'network'])
    """
    try:
        provider = input.provider.lower()
        resources = []
        
        # Import the provider module
        provider_module = importlib.import_module(f'diagrams.{provider}')
        
        if hasattr(provider_module, '__path__'):
            # Walk through all submodules in the provider
            for importer, modname, ispkg in pkgutil.iter_modules(provider_module.__path__):
                if not modname.startswith('_'):
                    resources.append(modname)
        
        resources.sort()
        logger.info(f"Found {len(resources)} resources for {provider}: {resources}")
        return resources
        
    except ImportError:
        logger.exception(f"Provider '{input.provider}' not found")
        return []
    except Exception as e:
        logger.exception(f"Failed to list resources for {input.provider}: {str(e)}")
        return []


@tool(args_schema=ListNodesByResourceInput)
def list_nodes_by_resource(input: ListNodesByResourceInput) -> List[str]:
    """List all available node classes for a specific provider and resource category.
    
    Args:
        input: Provider and resource specification
        
    Returns:
        List[str]: List of node class names (e.g., ['EC2', 'Lambda', 'ECS'])
    """
    try:
        provider = input.provider.lower()
        resource = input.resource.lower()
        nodes = []
        
        # Import the specific resource module
        module_name = f'diagrams.{provider}.{resource}'
        resource_module = importlib.import_module(module_name)
        
        # Get all classes from the module
        for name, obj in inspect.getmembers(resource_module, inspect.isclass):
            # Only include classes defined in this module (not imported ones)
            if obj.__module__ == module_name and not name.startswith('_'):
                nodes.append(name)
        
        nodes.sort()
        logger.info(f"Found {len(nodes)} nodes for {provider}.{resource}: {nodes}")
        return nodes
        
    except ImportError:
        logger.exception(f"Resource '{input.provider}.{input.resource}' not found")
        return []
    except Exception as e:
        logger.exception(f"Failed to list nodes for {input.provider}.{input.resource}: {str(e)}")
        return []


# Tool sets for different nodes
PLANNER_TOOLS = [
    list_all_providers,
    list_resources_by_provider,
    list_nodes_by_resource
]

EXECUTOR_TOOLS = [
    create_node,
    create_edge,
    create_cluster,
    add_to_graph,
    validate_graph,
    build_graph
]

# List of all available tools for easy import
ALL_GRAPH_TOOLS = [
    create_node,
    create_edge,
    create_cluster,
    build_graph,
    add_to_graph,
    validate_graph,
    generate_diagram,
    list_all_providers,
    list_resources_by_provider,
    list_nodes_by_resource
]