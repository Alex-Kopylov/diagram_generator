# MCP Graph Server

MCP (Model Context Protocol) server for graph construction tools.

## Installation

```bash
cd microservices/mcp_server
pip install -e .
```

## Usage

Run the MCP server:

```bash
python mcp_graph_server.py
```

## Available Tools

- `create_node(name, id=None)` - Create a new graph node
- `create_edge(source_id, target_id, forward=False, reverse=False)` - Create an edge between nodes
- `create_cluster(name, node_ids)` - Create a cluster containing specified nodes
- `build_graph(name, direction, nodes, edges, clusters=None)` - Build a complete graph
- `add_to_graph(graph, nodes=None, edges=None, clusters=None)` - Add components to existing graph
- `validate_graph(graph)` - Validate graph structure
- `graph_to_json(graph)` - Convert graph to JSON
- `generate_diagram(graph, output_file=None)` - Generate diagram from graph