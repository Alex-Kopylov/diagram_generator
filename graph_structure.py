from typing import Dict, List, Set, Any, Optional
import uuid


class Node:
    """Graph node with metadata"""
    
    @staticmethod
    def _rand_id():
        return uuid.uuid4().hex
    
    def __init__(self, node_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.id = node_id or self._rand_id()
        self.metadata = metadata or {}
    
    def __str__(self):
        return f"Node({self.id})"
    
    def __repr__(self):
        return f"Node(id='{self.id}', metadata={self.metadata})"
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


class Edge:
    """Graph edge with metadata"""
    
    @staticmethod
    def _rand_id():
        return uuid.uuid4().hex
    
    def __init__(self, source: Node, target: Node, metadata: Optional[Dict[str, Any]] = None):
        self.source = source
        self.target = target
        self.metadata = metadata or {}
        self.id = self._rand_id()
    
    def __str__(self):
        return f"Edge({self.source.id} -> {self.target.id})"
    
    def __repr__(self):
        return f"Edge(source={self.source.id}, target={self.target.id}, metadata={self.metadata})"
    
    def __eq__(self, other):
        return isinstance(other, Edge) and self.id == other.id
    
    def __hash__(self):
        return hash(self.id)


class Cluster:
    """Cluster of nodes"""
    
    @staticmethod
    def _rand_id():
        return uuid.uuid4().hex
    
    def __init__(self, name: str, nodes: Optional[Set[Node]] = None, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.nodes = nodes or set()
        self.metadata = metadata or {}
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
        return f"Cluster(name='{self.name}', nodes={len(self.nodes)}, metadata={self.metadata})"


class Graph:
    """Graph with name and metadata"""
    
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.metadata = metadata or {}
        
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
    
    def get_neighbors(self, node: Node) -> Set[Node]:
        """Get neighbors of node"""
        if node not in self._adjacency:
            return set()
        return {edge.target for edge in self._adjacency[node]}
    
    def get_edges_from(self, node: Node) -> Set[Edge]:
        """Get all edges from node"""
        return self._adjacency.get(node, set()).copy()
    
    def find_path(self, start: Node, end: Node) -> Optional[List[Node]]:
        """Find path between nodes (BFS)"""
        if start not in self._nodes or end not in self._nodes:
            return None
        
        if start == end:
            return [start]
        
        visited = set()
        queue = [(start, [start])]
        
        while queue:
            current, path = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
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


# Пример использования
if __name__ == "__main__":
    # Create graph
    graph = Graph("My Graph", {"description": "Example graph", "version": "1.0"})
    
    # Create nodes
    node1 = Node("A", {"type": "start", "value": 10})
    node2 = Node("B", {"type": "middle", "value": 20})
    node3 = Node("C", {"type": "end", "value": 30})
    
    # Add nodes to graph
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    
    # Create edges
    edge1 = Edge(node1, node2, {"weight": 1.5, "label": "first edge"})
    edge2 = Edge(node2, node3, {"weight": 2.0, "label": "second edge"})
    
    # Add edges to graph
    graph.add_edge(edge1)
    graph.add_edge(edge2)
    
    # Create cluster
    cluster1 = Cluster("Main Cluster", {node1, node2}, {"importance": "high"})
    graph.add_cluster(cluster1)
    
    # Graph information
    print(f"Graph: {graph}")
    print(f"Nodes: {graph.node_count()}")
    print(f"Edges: {graph.edge_count()}")
    print(f"Clusters: {graph.cluster_count()}")
    
    # Path finding
    path = graph.find_path(node1, node3)
    print(f"Path from A to C: {[n.id for n in path] if path else 'Not found'}")
    
    # Node neighbors
    neighbors = graph.get_neighbors(node1)
    print(f"Neighbors of node A: {[n.id for n in neighbors]}")
