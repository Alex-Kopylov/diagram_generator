# Diagram Generator Service

## Overview

An async, stateless Python API service that creates infrastructure diagrams using AI agents powered by Large Language Models (LLMs). Users describe diagram components, nodes, or flows in natural language and receive rendered PNG images.

The service uses a **LangGraph-based workflow** with separate planner and executor agents that leverage native tools built around the **diagrams** package for Python.

---

## Architecture

The service implements a **multi-agent workflow** using LangGraph:

* **Planner Agent**: Analyzes user requests and creates detailed execution plans using discovery tools
* **Executor Agent**: Executes plans by building graph structures using construction tools  
* **Graph Builder**: Generates final diagram images from completed graph structures

```mermaid
graph TD
    %% User Input
    A[User Request: Natural Language Description] --> B[DiagramAgent.generate_diagram]
    
    %% Main Workflow State
    B --> C[Initialize DiagramState]
    C --> D[Planner Node]
    
    %% Planner Agent Subgraph
    subgraph PlannerWorkflow ["Planner React Agent"]
        D --> D1[Planner Agent LLM]
        D1 --> D2{Tool Calls?}
        D2 -->|Yes| D3[Planner Tools Node]
        D2 -->|No| D4[Plan Generated]
        D3 --> D1
        
        %% Planner Tools
        subgraph PlannerTools ["Planner Tools"]
            PT1[list_all_providers]
            PT2[list_resources_by_provider]
            PT3[list_nodes_by_resource]
            PT4[validate_node_exists]
        end
        D3 --> PlannerTools
    end
    
    D4 --> E[Executor Node]
    
    %% Executor Agent Subgraph
    subgraph ExecutorWorkflow ["Executor React Agent"]
        E --> E1[Executor Agent LLM]
        E1 --> E2{Tool Calls?}
        E2 -->|Yes| E3[Executor Tools Node]
        E2 -->|No| E4[Graph Built]
        E3 --> E1
        
        %% Executor Tools
        subgraph ExecutorTools ["Executor Tools"]
            ET1[create_node]
            ET2[create_edge]
            ET3[create_cluster]
            ET4[create_empty_graph]
            ET5[add_node_to_graph]
            ET6[add_edge_to_graph]
            ET7[build_graph]
            ET8[validate_graph]
        end
        E3 --> ExecutorTools
    end
    
    E4 --> F[Graph Builder Node]
    F --> G[generate_diagram]
    G --> H[DiagramGenerationResult]
    
    %% Data Models
    subgraph DataModels ["Core Data Models"]
        DM1[Node: path, display_name, id]
        DM2[Edge: source, target, forward, reverse]
        DM3[Cluster: name, nodes]
        DM4[Graph: name, direction, nodes, edges, clusters]
    end
    
    %% State Management
    subgraph StateManagement ["State Management"]
        S1[DiagramState: message, plan, result, graph, success]
        S2[PlannerState: messages, plan]
        S3[ExecutorState: messages, graph]
    end
    
    %% External Dependencies
    subgraph External ["External Dependencies"]
        EX1[OpenAI GPT Models]
        EX2[Anthropic Claude Models]
        EX3[Diagrams Library]
        EX4[FastAPI Endpoints]
    end
    
    %% Connections to external systems
    D1 -.-> EX1
    D1 -.-> EX2
    E1 -.-> EX1
    E1 -.-> EX2
    G -.-> EX3
    H --> EX4
    
    %% Tool connections to data models
    ExecutorTools -.-> DataModels
    
    %% Styling
    classDef agent fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef tools fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef state fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef external fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class D,E,F agent
    class PlannerTools,ExecutorTools tools
    class StateManagement,DataModels state
    class External external
```

### Key Features

* **Native Tool Integration**: Custom tools that operate the diagrams package without exposing implementation details to the LLM
* **Stateless Operation**: No user sessions or persistent state required
* **LangGraph Workflow**: React-style agent pattern with conditional routing
* **Fallback Model Support**: Built-in fallback to alternative LLM models

---

## Library Abstraction Strategy

The service implements a **layered architecture** that completely abstracts the Python `diagrams` library complexity from LLMs:

### 1. Custom Data Models (`core/graph_structure.py`)
- **Pydantic Models**: Created `Node`, `Edge`, `Cluster`, `Graph` models that mirror but simplify diagrams library primitives
- **Type Safety**: Pydantic validation ensures data integrity throughout the workflow

### 2. Tool-Based Interface (`tools/graph_tools.py`)
The service provides **15+ native tools** split between discovery (planner) and construction (executor) phases. See [Tool Architecture](#tool-architecture) for complete listings.

### 3. Three-Stage Workflow
```
Planner (Discovery) → Executor (Construction) → Graph Builder (Rendering)
```

- **Planner Agent**: Uses discovery tools to explore available components and create execution plans
- **Executor Agent**: Uses construction tools to build internal graph representation following the plan
- **Graph Builder**: Converts internal `Graph` model to actual `diagrams` objects via `to_diagrams()` method

### 4. Custom React Agent Implementation

**Built from Scratch**: Instead of using LangGraph's prebuilt `create_react_agent()`, both planner and executor agents implement the **React pattern manually**:

```python
# Custom React loop implementation
def create_planner_agent_node():
    def planner_agent(state: PlannerState, config: RunnableConfig):
        llm = init_chat_model(...).bind_tools(PLANNER_TOOLS)
        response = llm.invoke(state["messages"], config)
        return {"messages": [response]}
    return planner_agent

# Conditional routing logic
def should_continue_planner(state: PlannerState):
    last_message = state["messages"][-1]
    return "continue" if last_message.tool_calls else "end"
```

---

## Requirements Met

✅ **Python + FastAPI**: Async framework with full OpenAPI documentation  
✅ **UV Package Management**: Modern Python package manager with lock files  
✅ **Stateless Service**: No database or session management required  
✅ **Docker + docker-compose**: Full containerization with health checks  
✅ **Custom Diagrams Tools**: 15+ native tools for graph construction and validation  
✅ **LLM Integration**: OpenAI GPT models with Anthropic Claude fallback  
✅ **Visible Prompt Logic**: No opaque framework calls.
⭐ **Multiple Node Types**: Support all of them.

---

## Setup and Installation

### Local Development

1. **Prerequisites**: Python 3.11+ and UV package manager
2. **Install dependencies**:
   ```bash
   cd microservices/diagram_agent
   uv sync
   ```

3. **Environment configuration**:
   ```bash
   # Copy and configure environment variables
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the service**:
   ```bash
   uv run python main.py
   ```

### Docker Deployment

1. **Build and run with docker-compose**:
   ```bash
   docker compose up --build
   ```

2. **Health check**:
   ```bash
   curl http://localhost:8000/health
   ```

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
MODEL_NAME=gpt-4.1
FALLBACK_MODEL_NAME=claude-3-5-sonnet

# Service Configuration  
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
RELOAD=false
TEMPERATURE=0.1
```

---

## API Endpoints

### Core Functionality

* **POST `/generate-diagram`**: Generate diagram from natural language description
  - Input: `{"message": "description of diagram"}`
  - Output: PNG image (binary response)

* **GET `/health`**: Service health check
* **GET `/`**: Service information and available endpoints
* **GET `/docs`**: Interactive API documentation

### Example Usage

```bash
# Generate a diagram
curl -X POST "http://localhost:8000/generate-diagram" \
  -H "Content-Type: application/json" \
  -d '{"message": "Create AWS web architecture with ALB, EC2, and RDS"}' \
  --output diagram.png
```

---

## Tool Architecture

### Planner Tools (Discovery)
- `list_all_providers`: Discover available cloud providers
- `list_resources_by_provider`: Find resource categories  
- `list_nodes_by_resource`: Get specific node types
- `validate_node_exists`: Verify node class availability

### Executor Tools (Construction)
- `create_node`, `create_edge`, `create_cluster`: Basic graph components
- `create_empty_graph`, `build_graph`: Graph structure management
- `add_node_to_graph`, `add_edge_to_graph`: Incremental building
- `validate_graph`: Structure validation
- `generate_diagram`: Final image generation

---

## Examples

### Example 1: Basic Web Application

**Input:**
```json
{
  "message": "Create a diagram showing a basic web application with an Application Load Balancer, two EC2 instances for the web servers, and an RDS database for storage. The web servers should be in a cluster named 'Web Tier'."
}
```

![Web Application Architecture](examples/real_example_1.png)

---

### Example 2: Microservices Architecture

**Input:**
```json
{
  "message": "Design a microservices architecture with three services: an authentication service, a payment service, and an order service. Include an API Gateway for routing, an SQS queue for message passing between services, and a shared RDS database. Group the services in a cluster called 'Microservices'. Add CloudWatch for monitoring."
}
```
![Microservices Architecture](examples/real_example_2.png)

---

## Project Structure

```
microservices/diagram_agent/
├── main.py              # FastAPI application entry point
├── agents/
│   ├── diagram_agent.py # LangGraph workflow implementation
│   └── prompts.py       # System prompts for planner/executor
├── api/
│   └── endpoints.py     # FastAPI route definitions
├── tools/
│   └── graph_tools.py   # Native diagram construction tools
├── core/
│   └── graph_structure.py # Graph data models
├── config/
│   ├── settings.py      # Configuration management
│   └── logger.py        # Logging setup
└── tests/               # Unit tests (NOT IMPLEMENTED)
```