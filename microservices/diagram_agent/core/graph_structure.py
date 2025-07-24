from typing import Dict, Set
from enum import Enum
import uuid
from pydantic import BaseModel, Field


class Direction(str, Enum):
    """Diagram layout directions."""
    TOP_BOTTOM = "TB"
    BOTTOM_TOP = "BT"
    LEFT_RIGHT = "LR"
    RIGHT_LEFT = "RL"


class Node(BaseModel):
    """Graph node"""
    
    name: str
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    
    def __str__(self):
        return f"Node({self.name})"
    
    def __repr__(self):
        return f"Node(name='{self.name}', id='{self.id}')"
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


class Edge(BaseModel):
    """Graph edge"""
    
    source: Node
    target: Node
    forward: bool = False
    reverse: bool = False
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    
    def __str__(self):
        return f"Edge({self.source.id} -> {self.target.id})"
    
    def __repr__(self):
        return f"Edge(source={self.source.id}, target={self.target.id})"
    
    def __eq__(self, other):
        return isinstance(other, Edge) and self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


class Cluster(BaseModel):
    """Cluster of nodes"""
    
    name: str
    nodes: Set[Node] = Field(default_factory=set)
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    
    def add_node(self, node: Node):
        """Add node to cluster"""
        self.nodes.add(node)
    
    def remove_node(self, node: Node):
        """Remove node from cluster"""
        self.nodes.discard(node)
    
    def has_node(self, node: Node) -> bool:
        """Check if cluster contains node"""
        return node in self.nodes
    
    def __str__(self):
        return f"Cluster({self.name})"
    
    def __repr__(self):
        return f"Cluster(name='{self.name}', nodes={len(self.nodes)})"


class Graph(BaseModel):
    """Graph with name"""
    
    name: str
    direction: Direction = Direction.TOP_BOTTOM
    nodes: Set[Node] = Field(default_factory=set)
    edges: Set[Edge] = Field(default_factory=set)
    clusters: Dict[str, Cluster] = Field(default_factory=dict)
    adjacency: Dict[Node, Set[Edge]] = Field(default_factory=dict, exclude=True)
    
    model_config = {"populate_by_name": True}
    
    def add_node(self, node: Node):
        """Add node to graph"""
        self.nodes.add(node)
        if node not in self.adjacency:
            self.adjacency[node] = set()
    
    def remove_node(self, node: Node):
        """Remove node from graph"""
        if node in self.nodes:
            # Remove all edges connected to this node
            edges_to_remove = [edge for edge in self.edges 
                             if edge.source == node or edge.target == node]
            for edge in edges_to_remove:
                self.remove_edge(edge)
            
            # Remove node from all clusters
            for cluster in self.clusters.values():
                cluster.remove_node(node)
            
            self.nodes.remove(node)
            self.adjacency.pop(node, None)
    
    def add_edge(self, edge: Edge):
        """Add edge to graph"""
        # Ensure nodes exist in graph
        self.add_node(edge.source)
        self.add_node(edge.target)
        
        self.edges.add(edge)
        self.adjacency[edge.source].add(edge)
        
    
    def remove_edge(self, edge: Edge):
        """Remove edge from graph"""
        if edge in self.edges:
            self.edges.remove(edge)
            self.adjacency[edge.source].discard(edge)
    
    def add_cluster(self, cluster: Cluster):
        """Add cluster to graph"""
        self.clusters[cluster.name] = cluster
        # Add all cluster nodes to graph
        for node in cluster.nodes:
            self.add_node(node)
    
    def remove_cluster(self, cluster_name: str):
        """Remove cluster from graph"""
        self.clusters.pop(cluster_name, None)
    
    def get_edges_from(self, node: Node) -> Set[Edge]:
        """Get all edges from node"""
        return self.adjacency.get(node, set()).copy()
    
    def node_count(self) -> int:
        """Number of nodes"""
        return len(self.nodes)
    
    def edge_count(self) -> int:
        """Number of edges"""
        return len(self.edges)
    
    def cluster_count(self) -> int:
        """Number of clusters"""
        return len(self.clusters)
    
    def __str__(self):
        return f"Graph({self.name})"
    
    def __repr__(self):
        return (f"Graph(name='{self.name}', direction='{self.direction}', nodes={self.node_count()}, "
                f"edges={self.edge_count()}, clusters={self.cluster_count()})")
    
    def to_diagrams(self):
        """Convert to mingrammer/diagrams format"""
        import importlib
        from diagrams import Diagram, Cluster as DiagramsCluster
        
        # Create diagram with direction
        with Diagram(self.name, direction=self.direction.value, show=False) as diagram:
            # Dictionary to store created nodes for edge connections
            diagram_nodes = {}
            cluster_nodes = {}
            
            # Create clusters first
            for cluster_name, cluster in self.clusters.items():
                with DiagramsCluster(cluster_name):
                    for node in cluster.nodes:
                        # Dynamically import and create node
                        module_path, class_name = node.name.rsplit('.', 1)
                        module = importlib.import_module(module_path)
                        node_class = getattr(module, class_name)
                        diagram_node = node_class(node.id)
                        diagram_nodes[node.id] = diagram_node
                        cluster_nodes[node.id] = diagram_node
            
            # Create standalone nodes (not in clusters)
            for node in self.nodes:
                if node.id not in diagram_nodes:
                    # Dynamically import and create node
                    module_path, class_name = node.name.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    node_class = getattr(module, class_name)
                    diagram_node = node_class(node.id)
                    diagram_nodes[node.id] = diagram_node
            
            # Create edges/connections
            for edge in self.edges:
                source_node = diagram_nodes[edge.source.id]
                target_node = diagram_nodes[edge.target.id]
                
                match (edge.forward, edge.reverse):
                    case (True, True):
                        # Bidirectional - create Edge with both directions
                        from diagrams import Edge as DiagramEdge
                        source_node - DiagramEdge(forward=True, reverse=True) - target_node
                    case (True, False):
                        # Forward direction
                        source_node >> target_node
                    case (False, True):
                        # Reverse direction  
                        source_node << target_node
                    case (False, False):
                        # No direction
                        source_node - target_node
        
        return diagram