from typing import Dict, List, Set, Any, Optional
import uuid


class Node:
    """Graph node"""
    
    @staticmethod
    def _rand_id():
        return uuid.uuid4().hex
    
    def __init__(self, node_id: Optional[str] = None):
        self.id = node_id or self._rand_id()
    
    def __str__(self):
        return f"Node({self.id})"
    
    def __repr__(self):
        return f"Node(id='{self.id}')"
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


class Edge:
    """Graph edge"""
    
    @staticmethod
    def _rand_id():
        return uuid.uuid4().hex
    
    def __init__(self, source: Node, target: Node):
        self.source = source
        self.target = target
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
    
    def __init__(self, name: str):
        self.name = name
        
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
        return (f"Graph(name='{self.name}', nodes={self.node_count()}, "
                f"edges={self.edge_count()}, clusters={self.cluster_count()})")


# Example usage
if __name__ == "__main__":
    # Create graph
    graph = Graph("My Graph")
    
    # Create nodes
    node1 = Node("A")
    node2 = Node("B")
    node3 = Node("C")
    
    # Add nodes to graph
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    
    # Create edges
    edge1 = Edge(node1, node2)
    edge2 = Edge(node2, node3)
    
    # Add edges to graph
    graph.add_edge(edge1)
    graph.add_edge(edge2)
    
    # Create cluster
    cluster1 = Cluster("Main Cluster", {node1, node2})
    graph.add_cluster(cluster1)
    
    # Graph information
    print(f"Graph: {graph}")
    print(f"Nodes: {graph.node_count()}")
    print(f"Edges: {graph.edge_count()}")
    print(f"Clusters: {graph.cluster_count()}")
    
    # Node edges
    edges = graph.get_edges_from(node1)
    print(f"Edges from node A: {[e.id for e in edges]}")
