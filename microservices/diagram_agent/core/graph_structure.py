import uuid
from enum import Enum

from pydantic import BaseModel, Field, field_serializer


class Direction(str, Enum):
    """Diagram layout directions.
    
    Defines the flow direction for diagram layout visualization.
    """
    TOP_BOTTOM = "TB"  # Vertical flow from top to bottom
    BOTTOM_TOP = "BT"  # Vertical flow from bottom to top
    LEFT_RIGHT = "LR"  # Horizontal flow from left to right
    RIGHT_LEFT = "RL"  # Horizontal flow from right to left


class Node(BaseModel):
    """Graph node representing a component in the diagram.
    
    Represents a single node/component that can be rendered in a diagram.
    Each node has a path to its corresponding diagrams library class and
    an optional display name for human-readable representation.
    """

    path: str = Field(
        description="Full import path to the diagrams class (e.g., 'diagrams.aws.compute.EC2')"
    )
    display_name: str | None = Field(
        default=None,
        description="Optional human-readable name for the node (e.g., 'Web Server')"
    )
    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for the node"
    )

    def __str__(self):
        return f"Node({self.display_name or self.path})"

    def __repr__(self):
        return f"Node(path='{self.path}', display_name='{self.display_name}', id='{self.id}')"

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Edge(BaseModel):
    """Graph edge representing a connection between two nodes.
    
    Defines a directed or undirected connection between two nodes with
    configurable direction indicators for visualization.
    """

    source: Node = Field(description="Source node of the connection")
    target: Node = Field(description="Target node of the connection")
    forward: bool = Field(
        default=False,
        description="Whether to show forward arrow (source -> target)"
    )
    reverse: bool = Field(
        default=False,
        description="Whether to show reverse arrow (source <- target)"
    )
    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for the edge"
    )

    def __str__(self):
        return f"Edge({self.source.id} -> {self.target.id})"

    def __repr__(self):
        return f"Edge(source={self.source.id}, target={self.target.id})"

    def __eq__(self, other):
        return isinstance(other, Edge) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class Cluster(BaseModel):
    """Cluster of nodes for logical grouping.
    
    Groups related nodes together for better organization and visualization.
    Clusters are rendered as containers/boxes around their contained nodes.
    """

    name: str = Field(description="Name/label of the cluster")
    nodes: set[Node] = Field(
        default_factory=set,
        description="Set of nodes contained within this cluster"
    )
    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for the cluster"
    )

    def add_node(self, node: Node):
        """Add a node to this cluster.
        
        Args:
            node: Node to add to the cluster
        """
        self.nodes.add(node)

    def remove_node(self, node: Node):
        """Remove a node from this cluster.
        
        Args:
            node: Node to remove from the cluster
        """
        self.nodes.discard(node)

    def has_node(self, node: Node) -> bool:
        """Check if this cluster contains a specific node.
        
        Args:
            node: Node to check for
            
        Returns:
            bool: True if node is in this cluster
        """
        return node in self.nodes

    def __str__(self):
        return f"Cluster({self.name})"

    def __repr__(self):
        return f"Cluster(name='{self.name}', nodes={len(self.nodes)})"

    @field_serializer('nodes')
    def serialize_nodes(self, nodes_set: set[Node]):
        """Serialize the nodes set to a list for JSON compatibility."""
        return [node.model_dump() for node in nodes_set]


class Graph(BaseModel):
    """Complete graph structure with nodes, edges, and clusters.
    
    Main container for the entire diagram structure. Contains all nodes,
    their connections (edges), logical groupings (clusters), and metadata
    for rendering the final diagram.
    """

    name: str = Field(description="Name/title of the diagram")
    direction: Direction = Field(
        default=Direction.TOP_BOTTOM,
        description="Layout direction for the diagram"
    )
    nodes: set[Node] = Field(
        default_factory=set,
        description="All nodes in the graph"
    )
    edges: set[Edge] = Field(
        default_factory=set,
        description="All connections between nodes"
    )
    clusters: dict[str, Cluster] = Field(
        default_factory=dict,
        description="Named clusters for grouping nodes"
    )
    adjacency: dict[Node, set[Edge]] = Field(
        default_factory=dict,
        exclude=True,
        description="Internal adjacency list for efficient edge lookups"
    )

    model_config = {
        "populate_by_name": True,  # Allow field population by alias
        "arbitrary_types_allowed": True,  # Allow complex types
        "use_enum_values": True  # Use enum values in serialization
    }

    def add_node(self, node: Node):
        """Add a node to the graph.
        
        Args:
            node: Node to add to the graph
        """
        self.nodes.add(node)
        if node not in self.adjacency:
            self.adjacency[node] = set()

    def remove_node(self, node: Node):
        """Remove a node from the graph.
        
        Also removes all edges connected to this node and removes it from all clusters.
        
        Args:
            node: Node to remove from the graph
        """
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
        """Add an edge to the graph.
        
        Automatically adds source and target nodes if they don't exist.
        
        Args:
            edge: Edge to add to the graph
        """
        # Ensure nodes exist in graph
        self.add_node(edge.source)
        self.add_node(edge.target)

        self.edges.add(edge)
        self.adjacency[edge.source].add(edge)


    def remove_edge(self, edge: Edge):
        """Remove an edge from the graph.
        
        Args:
            edge: Edge to remove from the graph
        """
        if edge in self.edges:
            self.edges.remove(edge)
            self.adjacency[edge.source].discard(edge)

    def add_cluster(self, cluster: Cluster):
        """Add a cluster to the graph.
        
        Automatically adds all cluster nodes to the graph.
        
        Args:
            cluster: Cluster to add to the graph
        """
        self.clusters[cluster.name] = cluster
        # Add all cluster nodes to graph
        for node in cluster.nodes:
            self.add_node(node)

    def remove_cluster(self, cluster_name: str):
        """Remove a cluster from the graph.
        
        Args:
            cluster_name: Name of the cluster to remove
        """
        self.clusters.pop(cluster_name, None)

    def get_edges_from(self, node: Node) -> set[Edge]:
        """Get all edges originating from a node.
        
        Args:
            node: Source node to get edges from
            
        Returns:
            Set[Edge]: All edges with this node as source
        """
        return self.adjacency.get(node, set()).copy()

    def node_count(self) -> int:
        """Get the total number of nodes in the graph.
        
        Returns:
            int: Number of nodes
        """
        return len(self.nodes)

    def edge_count(self) -> int:
        """Get the total number of edges in the graph.
        
        Returns:
            int: Number of edges
        """
        return len(self.edges)

    def cluster_count(self) -> int:
        """Get the total number of clusters in the graph.
        
        Returns:
            int: Number of clusters
        """
        return len(self.clusters)

    def __str__(self):
        return f"Graph({self.name})"

    def __repr__(self):
        return (f"Graph(name='{self.name}', direction='{self.direction}', nodes={self.node_count()}, "
                f"edges={self.edge_count()}, clusters={self.cluster_count()})")

    @field_serializer('nodes')
    def serialize_nodes(self, nodes_set: set[Node]):
        """Serialize the nodes set to a list for JSON compatibility."""
        return [node.model_dump() for node in nodes_set]

    @field_serializer('edges')
    def serialize_edges(self, edges_set: set[Edge]):
        """Serialize the edges set to a list for JSON compatibility."""
        return [edge.model_dump() for edge in edges_set]

    @field_serializer('clusters')
    def serialize_clusters(self, clusters_dict: dict[str, Cluster]):
        """Serialize the clusters dict with proper serialization."""
        return {k: v.model_dump() for k, v in clusters_dict.items()}

    def to_diagrams(self):
        """Convert graph to diagrams library format for rendering.
        
        Transforms the internal graph representation into the format
        expected by the diagrams library for actual diagram generation.
        
        Returns:
            Diagram: Configured diagrams.Diagram object ready for rendering
        """
        import importlib

        from diagrams import Cluster as DiagramsCluster
        from diagrams import Diagram

        # Create diagram with direction
        with Diagram(self.name, direction=self.direction.value if isinstance(self.direction, Direction) else self.direction, show=False) as diagram:
            # Dictionary to store created nodes for edge connections
            diagram_nodes = {}
            cluster_nodes = {}

            # Create clusters first
            for cluster_name, cluster in self.clusters.items():
                with DiagramsCluster(cluster_name):
                    for node in cluster.nodes:
                        # Dynamically import and create node
                        module_path, class_name = node.path.rsplit('.', 1)
                        module = importlib.import_module(module_path)
                        node_class = getattr(module, class_name)
                        label = node.display_name or node.id
                        diagram_node = node_class(label)
                        diagram_nodes[node.id] = diagram_node
                        cluster_nodes[node.id] = diagram_node

            # Create standalone nodes (not in clusters)
            for node in self.nodes:
                if node.id not in diagram_nodes:
                    # Dynamically import and create node
                    module_path, class_name = node.path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    node_class = getattr(module, class_name)
                    label = node.display_name or node.id
                    diagram_node = node_class(label)
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
