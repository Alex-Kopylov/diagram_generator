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


class ValidationResult(BaseModel):
    """Graph validation result schema."""
    valid: bool = Field(..., description="Whether the graph is valid")
    errors: List[str] = Field(..., description="List of validation errors")


class DiagramResult(BaseModel):
    """Diagram generation result schema."""
    success: bool = Field(..., description="Whether diagram generation succeeded")
    file_path: str | None = Field(None, description="Generated diagram file path")
    bytestring: bytes | None = Field(None, description="Generated diagram bytestring")
    error: str | None = Field(None, description="Error message if generation failed")


@tool(return_direct=True) 
def create_node(path: str, display_name: Optional[str] = None, id: Optional[str] = None) -> Node:
    """Create a new graph node.
    
    Args:
        path: Path to diagrams class (e.g., 'diagrams.aws.compute.EC2')
        display_name: Optional human-readable name for the node
        id: Optional custom node ID
        
    Returns:
        Node: Created node object
    """
    if id:
        return Node(path=path, display_name=display_name, id=id)
    else:
        return Node(path=path, display_name=display_name)


@tool(return_direct=True)
def create_edge(source_id: str, target_id: str, forward: bool = False, reverse: bool = False) -> Edge:
    """Create a new graph edge between nodes.
    
    Args:
        source_id: Source node ID
        target_id: Target node ID
        forward: Enable forward direction arrow
        reverse: Enable reverse direction arrow
        
    Returns:
        Edge: Created edge object
    """
    # Create dummy nodes for edge creation
    source_node = Node(path="temp", id=source_id)
    target_node = Node(path="temp", id=target_id)
    
    edge = Edge(
        source=source_node,
        target=target_node,
        forward=forward,
        reverse=reverse
    )
    
    return edge


@tool(return_direct=True)
def create_cluster(name: str, node_ids: List[str]) -> Cluster:
    """Create a cluster containing specified nodes.
    
    Args:
        name: Cluster name
        node_ids: List of node IDs to include in cluster
        
    Returns:
        Cluster: Created cluster object
    """
    cluster = Cluster(name=name)
    # Note: node_ids are provided but nodes need to be added separately
    # since we don't have access to the actual Node objects here
    return cluster


@tool(return_direct=True)
def create_empty_graph(name: str, direction: Direction = Direction.LEFT_RIGHT) -> Graph:
    """Create an empty graph structure.
    
    Args:
        name: Graph name
        direction: Graph layout direction
        
    Returns:
        Graph: Empty graph object
    """
    logger.debug(f"Creating empty graph: name={name}, direction={direction}")
    graph = Graph(name=name, direction=direction)
    logger.info(f"Empty graph created: {graph.name}")
    return graph


@tool(return_direct=True)
def build_graph(name: str, direction: Direction, nodes: List[Node], edges: List[Union[Edge, Dict[str, Any]]], clusters: Optional[List[Union[Cluster, Dict[str, Any]]]] = None) -> Graph:
    """Build a complete graph from components.
    
    Args:
        name: Graph name
        direction: Graph layout direction
        nodes: Array of node objects
        edges: Array of edge objects or edge data
        clusters: Array of cluster objects or cluster data
        
    Returns:
        Graph: Complete graph object
    """
    logger.debug(f"Building graph: name={name}, nodes={len(nodes)}, edges={len(edges)}")
    
    # Create graph
    graph = Graph(name=name, direction=direction)
    
    # Create and add nodes
    node_map = {}
    for node in nodes:
        graph.add_node(node)
        node_map[node.id] = node
        logger.debug(f"Added node to graph: {node.id}")
    
    # Create and add edges
    for edge_data in edges:
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
    if clusters:
        for cluster_data in clusters:
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


@tool(return_direct=True) 
def add_node_to_graph(graph: Graph, node: Node) -> Graph:
    """Add a single node to an existing graph.
    
    Args:
        graph: Existing graph object
        node: Node to add
        
    Returns:
        Graph: Updated graph object
    """
    graph.add_node(node)
    logger.info(f"Added node {node.id} to graph: {graph.name}")
    return graph


@tool(return_direct=True)
def add_edge_to_graph(graph: Graph, edge: Union[Edge, Dict[str, Any]]) -> Graph:
    """Add a single edge to an existing graph.
    
    Args:
        graph: Existing graph object
        edge: Edge to add
        
    Returns:
        Graph: Updated graph object
    """
    if isinstance(edge, dict):
        # Handle edge data with node IDs - need to find actual nodes
        node_map = {node.id: node for node in graph.nodes}
        if "source_id" in edge and "target_id" in edge:
            source_node = node_map.get(edge["source_id"])
            target_node = node_map.get(edge["target_id"])
            if source_node and target_node:
                edge_obj = Edge(
                    source=source_node,
                    target=target_node,
                    forward=edge.get("forward", False),
                    reverse=edge.get("reverse", False)
                )
                graph.add_edge(edge_obj)
                logger.info(f"Added edge {edge['source_id']} -> {edge['target_id']} to graph: {graph.name}")
        else:
            # Handle full edge data
            edge_obj = Edge.model_validate(edge)
            graph.add_edge(edge_obj)
            logger.info(f"Added edge to graph: {graph.name}")
    else:
        graph.add_edge(edge)
        logger.info(f"Added edge to graph: {graph.name}")
    
    return graph


@tool(return_direct=True)
def add_to_graph(graph: Graph, nodes: Optional[List[Node]] = None, edges: Optional[List[Union[Edge, Dict[str, Any]]]] = None, clusters: Optional[List[Union[Cluster, Dict[str, Any]]]] = None) -> Graph:
    """Add components to an existing graph.
    
    Args:
        graph: Existing graph object
        nodes: Nodes to add
        edges: Edges to add
        clusters: Clusters to add
        
    Returns:
        Graph: Updated graph object
    """
    logger.debug(f"Adding to graph: nodes={len(nodes or [])}, edges={len(edges or [])}")
    
    # Create node map for existing nodes
    node_map = {node.id: node for node in graph.nodes}
    
    # Add new nodes
    if nodes:
        for node in nodes:
            graph.add_node(node)
            node_map[node.id] = node
    
    # Add new edges
    if edges:
        for edge_data in edges:
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
    if clusters:
        for cluster_data in clusters:
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


@tool(return_direct=True)
def validate_graph(graph: Graph) -> ValidationResult:
    """Validate graph structure and connections.
    
    Args:
        graph: Graph to validate
        
    Returns:
        ValidationResult: Validation result with errors if any
    """
    logger.debug("Validating graph structure")
    
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




@tool(return_direct=True)
def generate_diagram(graph: Graph, output_file: Optional[str] = None) -> DiagramResult:
    """Generate diagram file from graph.
    
    Args:
        graph: Graph
        output_file: Optional output file name
        
    Returns:
        DiagramResult: Generation result with success status and file path
    """
    try:
        logger.debug(f"Generating diagram, output_file={output_file}")
        
        # Generate diagram using the existing to_diagrams method
        diagram = graph.to_diagrams()
        
        # Determine output file name
        if not output_file:
            output_file = f"{graph.name.lower().replace(' ', '_')}.png"
        
        logger.info(f"Diagram generated successfully: {output_file}")
        return DiagramResult(success=True, file_path=output_file, error=None,
                             bytestring=diagram._repr_png_())
    
    except Exception as e:
        logger.exception(f"Diagram generation failed: {str(e)}")
        return DiagramResult(success=False, file_path=None, error=str(e), bytestring=None)


@tool(return_direct=True)
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


@tool(return_direct=True)
def list_resources_by_provider(provider: str) -> List[str]:
    """List all resource categories for a specific provider.
    
    Args:
        provider: Provider name (e.g., 'aws', 'gcp', 'azure')
        
    Returns:
        List[str]: List of resource categories (e.g., ['compute', 'database', 'network'])
    """
    try:
        provider = provider.lower()
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
        logger.exception(f"Provider '{provider}' not found")
        return []
    except Exception as e:
        logger.exception(f"Failed to list resources for {provider}: {str(e)}")
        return []


@tool(return_direct=True)
def list_nodes_by_resource(provider: str, resource: str) -> List[str]:
    """List all available node classes for a specific provider and resource category.
    
    Args:
        provider: Provider name (e.g., 'aws', 'gcp', 'azure')
        resource: Resource category (e.g., 'compute', 'database', 'network')
        
    Returns:
        List[str]: List of node class names (e.g., ['EC2', 'Lambda', 'ECS'])
    """
    try:
        provider = provider.lower()
        resource = resource.lower()
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
        logger.exception(f"Resource '{provider}.{resource}' not found")
        return []
    except Exception as e:
        logger.exception(f"Failed to list nodes for {provider}.{resource}: {str(e)}")
        return []


@tool(return_direct=True)
def validate_node_exists(path: str) -> Dict[str, Any]:
    """Validate if a specific node class exists in the diagrams package.
    
    Args:
        path: Full path to the node class (e.g., 'diagrams.aws.database.Athena')
        
    Returns:
        Dict[str, Any]: Validation result with exists flag, alternatives if not found
    """
    try:
        # Parse the path
        parts = path.split('.')
        if len(parts) < 4 or parts[0] != 'diagrams':
            return {
                "exists": False, 
                "error": f"Invalid path format. Expected 'diagrams.provider.resource.NodeClass', got '{path}'",
                "alternatives": []
            }
        
        provider = parts[1].lower()
        resource = parts[2].lower()
        node_class = parts[3]
        
        # Try to import the module and check if the class exists
        module_name = f'diagrams.{provider}.{resource}'
        try:
            resource_module = importlib.import_module(module_name)
            
            # Check if the specific class exists
            if hasattr(resource_module, node_class):
                logger.info(f"Node class validated: {path}")
                return {
                    "exists": True,
                    "path": path,
                    "alternatives": []
                }
            else:
                # Get available alternatives in the same resource
                alternatives = []
                for name, obj in inspect.getmembers(resource_module, inspect.isclass):
                    if obj.__module__ == module_name and not name.startswith('_'):
                        alternatives.append(f"diagrams.{provider}.{resource}.{name}")
                
                logger.warning(f"Node class '{node_class}' not found in {module_name}. Available: {alternatives}")
                return {
                    "exists": False,
                    "error": f"Node class '{node_class}' not found in '{module_name}'",
                    "alternatives": alternatives[:5]  # Limit to 5 alternatives
                }
                
        except ImportError:
            logger.warning(f"Module '{module_name}' not found")
            return {
                "exists": False,
                "error": f"Module '{module_name}' not found",
                "alternatives": []
            }
            
    except Exception as e:
        logger.exception(f"Failed to validate node {path}: {str(e)}")
        return {
            "exists": False,
            "error": f"Validation failed: {str(e)}",
            "alternatives": []
        }


# Tool sets for different nodes
PLANNER_TOOLS = [
    list_all_providers,
    list_resources_by_provider,
    list_nodes_by_resource,
    validate_node_exists
]

EXECUTOR_TOOLS = [
    create_node,
    create_edge,
    create_cluster,
    create_empty_graph,
    add_node_to_graph,
    add_edge_to_graph,
    add_to_graph,
    validate_graph,
    build_graph
]

# List of all available tools for easy import
ALL_GRAPH_TOOLS = [
    create_node,
    create_edge,
    create_cluster,
    create_empty_graph,
    add_node_to_graph,
    add_edge_to_graph,
    build_graph,
    add_to_graph,
    validate_graph,
    generate_diagram,
    list_all_providers,
    list_resources_by_provider,
    list_nodes_by_resource
]