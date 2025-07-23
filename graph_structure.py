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
    _nodes: Set[Node] = Field(default_factory=set, alias="nodes")
    _edges: Set[Edge] = Field(default_factory=set, alias="edges")
    _clusters: Dict[str, Cluster] = Field(default_factory=dict, alias="clusters")
    _adjacency: Dict[Node, Set[Edge]] = Field(default_factory=dict, exclude=True)
    
    model_config = {"populate_by_name": True}
    
    def add_node(self, node: Node):
        """Add node to graph"""
        self._nodes.add(node)
        if node not in self._adjacency:
            self._adjacency[node] = set()
    
    def remove_node(self, node: Node):
        """Remove node from graph"""
        if node in self._nodes:
            # Remove all edges connected to this node
            edges_to_remove = [edge for edge in self._edges 
                             if edge.source == node or edge.target == node]
            for edge in edges_to_remove:
                self.remove_edge(edge)
            
            # Remove node from all clusters
            for cluster in self._clusters.values():
                cluster.remove_node(node)
            
            self._nodes.remove(node)
            self._adjacency.pop(node, None)
    
    def add_edge(self, edge: Edge):
        """Add edge to graph"""
        # Ensure nodes exist in graph
        self.add_node(edge.source)
        self.add_node(edge.target)
        
        self._edges.add(edge)
        self._adjacency[edge.source].add(edge)
        
    
    def remove_edge(self, edge: Edge):
        """Remove edge from graph"""
        if edge in self._edges:
            self._edges.remove(edge)
            self._adjacency[edge.source].discard(edge)
    
    def add_cluster(self, cluster: Cluster):
        """Add cluster to graph"""
        self._clusters[cluster.name] = cluster
        # Add all cluster nodes to graph
        for node in cluster.nodes:
            self.add_node(node)
    
    def remove_cluster(self, cluster_name: str):
        """Remove cluster from graph"""
        self._clusters.pop(cluster_name, None)
    
    def get_edges_from(self, node: Node) -> Set[Edge]:
        """Get all edges from node"""
        return self._adjacency.get(node, set()).copy()
    
    @property
    def nodes(self) -> Set[Node]:
        """Get all nodes in graph"""
        return self._nodes.copy()
    
    @property
    def edges(self) -> Set[Edge]:
        """Get all edges in graph"""
        return self._edges.copy()
    
    @property
    def clusters(self) -> Dict[str, Cluster]:
        """Get all clusters in graph"""
        return self._clusters.copy()
    
    def node_count(self) -> int:
        """Number of nodes"""
        return len(self._nodes)
    
    def edge_count(self) -> int:
        """Number of edges"""
        return len(self._edges)
    
    def cluster_count(self) -> int:
        """Number of clusters"""
        return len(self._clusters)
    
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
            for cluster_name, cluster in self._clusters.items():
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
            for node in self._nodes:
                if node.id not in diagram_nodes:
                    # Dynamically import and create node
                    module_path, class_name = node.name.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    node_class = getattr(module, class_name)
                    diagram_node = node_class(node.id)
                    diagram_nodes[node.id] = diagram_node
            
            # Create edges/connections
            for edge in self._edges:
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
    


# Test: Microservices Architecture Simulation
if __name__ == "__main__":
    # Create microservices architecture graph
    graph = Graph(name="Microservices Architecture", direction=Direction.LEFT_RIGHT)
    
    # Create nodes for components (using node_id parameter instead of second positional arg)
    api_gateway = Node(name="diagrams.aws.network.APIGateway", id="API Gateway")
    auth_service = Node(name="diagrams.aws.compute.EC2", id="Auth Service")
    payment_service = Node(name="diagrams.aws.compute.EC2", id="Payment Service")
    order_service = Node(name="diagrams.aws.compute.EC2", id="Order Service")
    sqs_queue = Node(name="diagrams.aws.integration.SQS", id="SQS Queue")
    database = Node(name="diagrams.aws.database.RDS", id="Shared RDS")
    monitoring = Node(name="diagrams.aws.management.Cloudwatch", id="Monitoring")
    
    # Add all nodes to graph
    for node in [api_gateway, auth_service, payment_service, order_service, sqs_queue, database, monitoring]:
        graph.add_node(node)
    
    # Create microservices cluster
    microservices_cluster = Cluster(name="Microservices")
    microservices_cluster.add_node(auth_service)
    microservices_cluster.add_node(payment_service)
    microservices_cluster.add_node(order_service)
    graph.add_cluster(microservices_cluster)
    
    # Create edges (connections) - API Gateway to services
    graph.add_edge(Edge(source=api_gateway, target=auth_service, forward=True))
    graph.add_edge(Edge(source=api_gateway, target=payment_service, forward=True))
    graph.add_edge(Edge(source=api_gateway, target=order_service, forward=True))
    
    # Services to SQS queue
    graph.add_edge(Edge(source=auth_service, target=sqs_queue, forward=True))
    graph.add_edge(Edge(source=payment_service, target=sqs_queue, forward=True))
    graph.add_edge(Edge(source=order_service, target=sqs_queue, forward=True))
    
    # Services to database
    graph.add_edge(Edge(source=auth_service, target=database, forward=True))
    graph.add_edge(Edge(source=payment_service, target=database, forward=True))
    graph.add_edge(Edge(source=order_service, target=database, forward=True))
    
    # Monitoring to services
    graph.add_edge(Edge(source=monitoring, target=auth_service, forward=True))
    graph.add_edge(Edge(source=monitoring, target=payment_service, forward=True))
    graph.add_edge(Edge(source=monitoring, target=order_service, forward=True))
    
    # Display graph information
    print(f"Graph: {graph}")
    print(f"Nodes: {graph.node_count()}")
    print(f"Edges: {graph.edge_count()}")
    print(f"Clusters: {graph.cluster_count()}")
    print(f"Microservices cluster nodes: {len(microservices_cluster.nodes)}")
    
    # Test edge connectivity
    api_edges = graph.get_edges_from(api_gateway)
    print(f"API Gateway connects to {len(api_edges)} services")
    
    # Test JSON serialization
    json_data = graph.model_dump_json(indent=2)
    print("JSON serialization works!")
    
    # Test JSON deserialization
    graph_from_json = Graph.model_validate_json(json_data)
    print(f"Deserialized graph: {graph_from_json}")
    
    graph.to_diagrams()
    print("Microservices architecture simulation completed!")