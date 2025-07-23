# Diagram Generation Examples

This directory contains examples demonstrating the intelligent agent-based approach to diagram generation using the `diagrams` library. These examples show how natural language descriptions are transformed into structured tool calls and executed to create infrastructure diagrams.

## Examples Overview

### 1. Basic Web Application (`basic_web_application.py`)
**Natural Language Description:**
> "Create a diagram showing a basic web application with an Application Load Balancer, two EC2 instances for the web servers, and an RDS database for storage. The web servers should be in a cluster named 'Web Tier'."

**Generated Diagram:**
- Application Load Balancer (AWS ELB)
- Web Tier cluster containing two EC2 instances
- RDS database for storage
- Connections showing traffic flow

**Key Features:**
- Demonstrates cluster usage for logical grouping
- Shows load balancer distribution pattern
- Illustrates database connectivity

### 2. Microservices Architecture (`microservices_architecture.py`)
**Natural Language Description:**
> "Design a microservices architecture with three services: an authentication service, a payment service, and an order service. Include an API Gateway for routing, an SQS queue for message passing between services, and a shared RDS database. Group the services in a cluster called 'Microservices'. Add CloudWatch for monitoring."

**Generated Diagram:**
- API Gateway for request routing
- Microservices cluster with three services
- SQS queue for inter-service messaging
- Shared RDS database
- CloudWatch monitoring

**Key Features:**
- Multiple service types and interactions
- Message queue integration
- Monitoring and observability
- Complex multi-service relationships

### 3. LLM Tool Call Simulation (`llm_tool_call_simulation.py`)
**Purpose:**
Demonstrates how an intelligent agent would break down natural language descriptions into structured tool calls for diagram generation.

**What it Shows:**
- Complete tool execution sequence for both examples
- Agent planning and reasoning process
- Tool call parameters and expected responses
- Error handling and validation
- Final code generation from tool calls

**Tool Types Demonstrated:**
- `CREATE_DIAGRAM` - Initialize diagram context
- `CREATE_NODE` - Create individual components
- `CREATE_CLUSTER` - Group related components
- `CREATE_EDGE` - Connect components with relationships
- `SEARCH_NODE` - Find available node types

## Running the Examples

### Prerequisites
```bash
# Install dependencies using UV
uv sync

# Or install manually
pip install diagrams
```

### Execute Examples
```bash
# Run basic web application example
python basic_web_application.py

# Run microservices architecture example
python microservices_architecture.py

# Run LLM tool call simulation
python llm_tool_call_simulation.py
```

### Generated Output
Each example creates a PNG file in the current directory:
- `basic_web_app.png` - Basic web application diagram
- `microservices_arch.png` - Microservices architecture diagram

## Agent Tool Architecture

The LLM tool call simulation demonstrates how the FastAPI service would implement an intelligent agent that:

1. **Parses** natural language descriptions
2. **Plans** the diagram structure and components
3. **Executes** tool calls in the correct sequence
4. **Validates** tool responses and handles errors
5. **Generates** the final diagram code

### Available Tools

#### 1. Create Diagram Tool
```python
{
    "name": "Basic Web Application",
    "outformat": "png",
    "filename": "basic_web_app",
    "show": False,
    "direction": "TB"
}
```

#### 2. Create Node Tool
```python
{
    "provider": "aws",
    "resource_type": "compute",
    "node_class": "EC2",
    "label": "Web Server 1",
    "cluster": "Web Tier"
}
```

#### 3. Create Cluster Tool
```python
{
    "name": "Web Tier",
    "parent_cluster": None,
    "graph_attr": {"style": "filled", "color": "lightblue"}
}
```

#### 4. Create Edge Tool
```python
{
    "from_node": "alb_001",
    "to_node": ["web_server_1", "web_server_2"],
    "direction": ">>",
    "label": "HTTP Traffic",
    "color": "blue",
    "style": "solid"
}
```

#### 5. Search Node Tool
```python
{
    "query": "API Gateway",
    "provider": "aws",
    "category": "network"
}
```

## Integration with FastAPI Service

These examples demonstrate the core functionality that would be implemented in the FastAPI service:

1. **POST /diagram** endpoint would:
   - Accept natural language description
   - Use LLM to generate tool call sequence
   - Execute tools to create diagram
   - Return PNG/SVG image

2. **POST /assistant** endpoint would:
   - Maintain conversation context
   - Ask clarifying questions
   - Provide explanations of the diagram creation process
   - Handle multi-turn interactions

## Key Design Patterns

### 1. Tool-Based Architecture
- Each diagram operation is a discrete tool
- Tools have well-defined parameters and responses
- Agent orchestrates tool execution
- Error handling at each tool level

### 2. Hierarchical Planning
- High-level plan from natural language
- Detailed tool sequence generation
- Dependency management between tools
- Validation and error recovery

### 3. Multi-Cloud Support
- AWS, GCP, Azure node types
- Provider-agnostic tool interface
- Extensible node search and discovery
- Consistent tool responses across providers

## Error Handling Examples

The simulation includes examples of:
- Invalid node type searches
- Cluster dependency validation
- Edge connection verification
- Resource naming conflicts
- Tool parameter validation

## Next Steps

To implement the full FastAPI service:

1. Create tool implementations for each tool type
2. Integrate with OpenAI API for LLM orchestration
3. Add proper error handling and validation
4. Implement image generation and response formatting
5. Add comprehensive test coverage with mocked LLM calls
6. Create Docker containerization
7. Add monitoring and logging

## Additional Resources

- [Diagrams Documentation](https://diagrams.mingrammer.com/)
- [AWS Node Types](https://diagrams.mingrammer.com/docs/nodes/aws)
- [GCP Node Types](https://diagrams.mingrammer.com/docs/nodes/gcp)
- [Azure Node Types](https://diagrams.mingrammer.com/docs/nodes/azure)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference) 