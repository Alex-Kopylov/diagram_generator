from typing import Dict, List, Set, Any, Optional
from enum import Enum
import uuid


class Direction(str, Enum):
    """Diagram layout directions."""
    TOP_BOTTOM = "TB"
    BOTTOM_TOP = "BT"
    LEFT_RIGHT = "LR"
    RIGHT_LEFT = "RL"


class Node:
    """Graph node"""
    
    @staticmethod
    def _rand_id():
        return uuid.uuid4().hex
    
    def __init__(self, name: str, node_id: Optional[str] = None):
        self.name = name
        self.id = node_id or self._rand_id()
    
    def __str__(self):
        return f"Node({self.name})"
    
    def __repr__(self):
        return f"Node(name='{self.name}', id='{self.id}')"
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


class Edge:
    """Graph edge"""
    
    @staticmethod
    def _rand_id():
        return uuid.uuid4().hex
    
    def __init__(self, source: Node, target: Node, forward: bool = False, reverse: bool = False):
        self.source = source
        self.target = target
        self.forward = forward
        self.reverse = reverse
        self.id = self._rand_id()
    
    def __str__(self):
        return f"Edge({self.source.id} -> {self.target.id})"
    
    def __repr__(self):
        return f"Edge(source={self.source.id}, target={self.target.id})"
    
    def __eq__(self, other):
        return isinstance(other, Edge) and self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


class Cluster:
    """Cluster of nodes"""
    
    @staticmethod
    def _rand_id():
        return uuid.uuid4().hex
    
    def __init__(self, name: str, nodes: Optional[Set[Node]] = None):
        self.name = name
        self.nodes = nodes or set()
        self.id = self._rand_id()
    
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


class Graph:
    """Graph with name"""
    
    def __init__(self, name: str, direction: Direction = Direction.TOP_BOTTOM):
        self.name = name
        self.direction = direction
        
        self._nodes: Set[Node] = set()
        self._edges: Set[Edge] = set()
        self._clusters: Dict[str, Cluster] = {}
        
        # For fast edge lookup
        self._adjacency: Dict[Node, Set[Edge]] = {}
    
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
                
                if edge.forward and edge.reverse:
                    # Bidirectional
                    source_node - target_node
                elif edge.forward:
                    # Forward direction
                    source_node >> target_node
                elif edge.reverse:
                    # Reverse direction
                    source_node << target_node
                else:
                    # Default connection
                    source_node - target_node
        
        return diagram


# Test: Microservices Architecture Simulation
if __name__ == "__main__":
    # Create microservices architecture graph
    graph = Graph("Microservices Architecture", Direction.LEFT_RIGHT)
    
    # Create nodes for components
    api_gateway = Node("diagrams.aws.network.APIGateway", "API Gateway")
    auth_service = Node("diagrams.aws.compute.EC2", "Auth Service")
    payment_service = Node("diagrams.aws.compute.EC2", "Payment Service")
    order_service = Node("diagrams.aws.compute.EC2", "Order Service")
    sqs_queue = Node("diagrams.aws.integration.SQS", "SQS Queue")
    database = Node("diagrams.aws.database.RDS", "Shared RDS")
    monitoring = Node("diagrams.aws.management.Cloudwatch", "Monitoring")
    
    # Add all nodes to graph
    for node in [api_gateway, auth_service, payment_service, order_service, sqs_queue, database, monitoring]:
        graph.add_node(node)
    
    # Create microservices cluster
    microservices_cluster = Cluster("Microservices")
    microservices_cluster.add_node(auth_service)
    microservices_cluster.add_node(payment_service)
    microservices_cluster.add_node(order_service)
    graph.add_cluster(microservices_cluster)
    
    # Create edges (connections) - API Gateway to services
    graph.add_edge(Edge(api_gateway, auth_service, forward=True))
    graph.add_edge(Edge(api_gateway, payment_service, forward=True))
    graph.add_edge(Edge(api_gateway, order_service, forward=True))
    
    # Services to SQS queue
    graph.add_edge(Edge(auth_service, sqs_queue, forward=True))
    graph.add_edge(Edge(payment_service, sqs_queue, forward=True))
    graph.add_edge(Edge(order_service, sqs_queue, forward=True))
    
    # Services to database
    graph.add_edge(Edge(auth_service, database, forward=True))
    graph.add_edge(Edge(payment_service, database, forward=True))
    graph.add_edge(Edge(order_service, database, forward=True))
    
    # Monitoring to services
    graph.add_edge(Edge(monitoring, auth_service, forward=True))
    graph.add_edge(Edge(monitoring, payment_service, forward=True))
    graph.add_edge(Edge(monitoring, order_service, forward=True))
    
    # Display graph information
    print(f"Graph: {graph}")
    print(f"Nodes: {graph.node_count()}")
    print(f"Edges: {graph.edge_count()}")
    print(f"Clusters: {graph.cluster_count()}")
    print(f"Microservices cluster nodes: {len(microservices_cluster.nodes)}")
    
    # Test edge connectivity
    api_edges = graph.get_edges_from(api_gateway)
    print(f"API Gateway connects to {len(api_edges)} services")
    graph.to_diagrams()
    print("Microservices architecture simulation completed!")