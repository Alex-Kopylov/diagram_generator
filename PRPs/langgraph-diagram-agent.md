# PRP: LangGraph Diagram Agent

## Overview

Create an intelligent LangGraph agent that uses the existing MCP graph server to analyze user messages, plan tool call sequences, execute graph construction, and generate diagrams. The agent should have two modes: stateless completion (primary focus) and chat with memory (placeholder implementation).

## Feature Requirements

### Core Functionality
- **LangGraph Agent** with intelligent workflow orchestration
- **4-Node Architecture**:
  1. **Analyze Node**: Parse and understand user message intent
  2. **Plan Node**: Determine optimal tool call sequence  
  3. **Execute Node**: Call MCP tools and build graph
  4. **Output Node**: Generate final diagram using `Graph.to_diagrams()`
- **Two Operation Modes**:
  - **Stateless Completion**: Single-shot diagram generation
  - **Chat with Memory**: Session-based interaction (placeholder with `NotImplementedError`)
- **MCP Integration**: Use existing `mcp_graph_server.py` tools
- **FastAPI Endpoints**: RESTful API with proper error handling

## Research Context

### LangGraph Modern Patterns (v0.5.0+)
- **Documentation**: https://langchain-ai.github.io/langgraph/concepts/low_level/
- **StateGraph Reference**: https://langchain-ai.github.io/langgraph/reference/graphs/
- **MCP Integration**: https://langchain-ai.github.io/langgraph/agents/mcp/

### Key Technical Patterns
1. **TypedDict with Annotated Reducers**:
   ```python
   from typing import Annotated
   from langgraph.graph.message import add_messages
   
   class AgentState(TypedDict):
       messages: Annotated[list[AnyMessage], add_messages]
       user_intent: str
       tool_plan: list[dict]
       graph: Optional[Graph]
       result: Optional[Any]
   ```

2. **MCP Multi-Server Integration**:
   ```python
   from langchain_mcp_adapters.client import MultiServerMCPClient
   
   client = MultiServerMCPClient({
       "graph": {
           "command": "python", 
           "args": ["./mcp_graph_server.py"],
           "transport": "stdio"
       }
   })
   ```

3. **Session Management**:
   ```python
   # Stateless (primary)
   graph = builder.compile()  # No checkpointer
   
   # Stateful (placeholder)  
   from langgraph.checkpoint.memory import MemorySaver
   checkpointer = MemorySaver()
   graph = builder.compile(checkpointer=checkpointer)  
   ```

## Architecture Design

### State Schema
```python
class DiagramAgentState(TypedDict):
    """LangGraph state for diagram agent workflow."""
    messages: Annotated[list[BaseMessage], add_messages]
    user_intent: str  # Parsed user requirements
    tool_plan: list[dict]  # Planned MCP tool sequence
    graph: Optional[Graph]  # Built graph object
    diagram_result: Optional[Any]  # Final diagram output
    error: Optional[str]  # Error handling
```

### Node Functions
1. **analyze_user_message**: Parse intent, extract requirements
2. **plan_tool_sequence**: Determine optimal MCP tool call order
3. **execute_graph_construction**: Call MCP tools, build graph
4. **generate_diagram_output**: Call `Graph.to_diagrams()`, return result

### Flow Architecture
```
START → analyze_user_message → plan_tool_sequence → execute_graph_construction → generate_diagram_output → END
```

## Implementation Blueprint

### File Structure
```
microservices/
├── diagram_agent/
│   ├── agents/
│   │   ├── langgraph_agent.py      # Main LangGraph agent implementation
│   │   ├── nodes.py                # Individual node functions  
│   │   ├── state.py                # State schema definitions
│   │   └── prompts.py              # Enhanced system prompts
│   ├── api/
│   │   ├── endpoints.py            # FastAPI route handlers
│   │   └── schemas.py              # Request/response models
│   ├── tools/
│   │   └── mcp_client.py           # MCP client wrapper
│   └── pyproject.toml              # Diagram agent dependencies
└── mcp_server/
    ├── mcp_graph_server.py         # MCP server implementation
    ├── graph_structure.py          # Graph data structures
    └── pyproject.toml              # MCP server dependencies
```

### Key Components

#### 1. LangGraph Agent (`agents/langgraph_agent.py`)
```python
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.client import MultiServerMCPClient

class DiagramAgent:
    def __init__(self):
        self.mcp_client = self._setup_mcp_client()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> CompiledGraph:
        builder = StateGraph(DiagramAgentState)
        
        builder.add_node("analyze", analyze_user_message)
        builder.add_node("plan", plan_tool_sequence) 
        builder.add_node("execute", execute_graph_construction)
        builder.add_node("output", generate_diagram_output)
        
        builder.set_entry_point("analyze")
        builder.add_edge("analyze", "plan")
        builder.add_edge("plan", "execute") 
        builder.add_edge("execute", "output")
        builder.add_edge("output", END)
        
        return builder.compile()  # Stateless by default
```

#### 2. Node Implementations (`agents/nodes.py`)
```python
async def analyze_user_message(state: DiagramAgentState) -> DiagramAgentState:
    """Analyze user message to extract diagram requirements."""
    # Use LLM to parse intent, identify components, relationships
    # Return updated state with user_intent field populated
    
async def plan_tool_sequence(state: DiagramAgentState) -> DiagramAgentState:
    """Plan optimal sequence of MCP tool calls."""
    # Analyze requirements, determine tool call order
    # Return state with tool_plan populated
    
async def execute_graph_construction(state: DiagramAgentState) -> DiagramAgentState:
    """Execute MCP tool calls to build graph."""
    # Call mcp_graph_server tools in planned sequence
    # Return state with completed graph object
    
async def generate_diagram_output(state: DiagramAgentState) -> DiagramAgentState:
    """Generate final diagram using Graph.to_diagrams()."""
    # Call graph.to_diagrams(), return Diagram object
    # Handle any conversion errors
```

#### 3. FastAPI Integration (`api/endpoints.py`)
```python
from fastapi import FastAPI
from diagram_agent.agents.langgraph_agent import DiagramAgent

app = FastAPI(title="LangGraph Diagram Agent")
agent = DiagramAgent()

@app.post("/generate-diagram")
async def generate_diagram(request: DiagramRequest):
    """Stateless diagram generation endpoint."""
    result = await agent.generate_stateless(request.message)
    return DiagramResponse(**result)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint - placeholder implementation."""
    raise NotImplementedError("Chat mode will be implemented in future iteration")
```

## MCP Integration Strategy

### MCP Client Setup
```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "graph_tools": {
        "command": "python",
        "args": ["../mcp_server/mcp_graph_server.py"],
        "transport": "stdio"
    }
})

# Available tools from mcp_graph_server.py:
# - create_node(name, id=None)
# - create_edge(source_id, target_id, forward=False, reverse=False)  
# - create_cluster(name, node_ids)
# - build_graph(name, direction, nodes, edges, clusters=None)
# - generate_diagram(graph, output_file=None)
```

### Tool Execution Pattern
```python
async def execute_mcp_tools(tool_plan: list[dict], mcp_client) -> Graph:
    """Execute planned MCP tool sequence."""
    graph = None
    
    for tool_call in tool_plan:
        tool_name = tool_call["tool"]
        tool_args = tool_call["args"]
        
        if tool_name == "build_graph":
            graph = await mcp_client.call_tool("build_graph", **tool_args)
        elif tool_name == "generate_diagram":
            diagram = await mcp_client.call_tool("generate_diagram", graph=graph, **tool_args)
            return diagram
    
    return graph
```

## Error Handling Strategy

### Validation Patterns
```python
def validate_user_intent(state: DiagramAgentState) -> DiagramAgentState:
    """Validate parsed user intent is actionable."""
    if not state.get("user_intent"):
        state["error"] = "Could not parse user requirements"
    return state

def validate_tool_plan(state: DiagramAgentState) -> DiagramAgentState:
    """Validate tool plan is executable."""
    required_tools = ["create_node", "build_graph", "generate_diagram"]
    plan_tools = [step["tool"] for step in state.get("tool_plan", [])]
    
    if not any(tool in plan_tools for tool in required_tools):
        state["error"] = "Invalid tool sequence planned"
    return state
```

### Retry and Fallback
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def robust_mcp_call(tool_name: str, **kwargs):
    """MCP tool call with retry logic."""
    try:
        return await mcp_client.call_tool(tool_name, **kwargs)
    except MCPError as e:
        logger.warning(f"MCP call failed: {e}")
        raise
```

## Testing Strategy

### Unit Tests Structure
```
tests/
├── unit/
│   ├── test_agent_nodes.py         # Test individual node functions
│   ├── test_state_management.py    # Test state transitions
│   ├── test_mcp_integration.py     # Test MCP client integration
│   └── test_diagram_generation.py  # Test output generation
├── integration/
│   ├── test_full_workflow.py       # End-to-end workflow tests
│   └── test_api_endpoints.py       # FastAPI endpoint tests
└── fixtures/
    ├── sample_requests.json        # Test input data
    └── expected_graphs.json        # Expected outputs
```

### Key Test Cases
```python
# tests/unit/test_agent_nodes.py
@pytest.mark.asyncio
async def test_analyze_user_message():
    """Test user message analysis node."""
    state = DiagramAgentState(
        messages=[HumanMessage("Create a web app with load balancer and database")]
    )
    
    result = await analyze_user_message(state)
    
    assert "user_intent" in result
    assert "load balancer" in result["user_intent"].lower()
    assert "database" in result["user_intent"].lower()

@pytest.mark.asyncio 
async def test_plan_tool_sequence():
    """Test tool sequence planning."""
    state = DiagramAgentState(
        user_intent="Create web architecture with ALB, EC2, RDS"
    )
    
    result = await plan_tool_sequence(state)
    
    assert "tool_plan" in result
    assert len(result["tool_plan"]) > 0
    assert result["tool_plan"][-1]["tool"] == "generate_diagram"
```

## Implementation Tasks

### Phase 1: Core Infrastructure
1. **Set up LangGraph StateGraph with DiagramAgentState**
2. **Implement MCP client wrapper for graph server integration**  
3. **Create basic node function stubs**
4. **Set up FastAPI application with health endpoint**

### Phase 2: Node Implementation
5. **Implement analyze_user_message node with LLM-based parsing**
6. **Implement plan_tool_sequence node with tool planning logic**
7. **Implement execute_graph_construction with MCP tool orchestration**
8. **Implement generate_diagram_output with Graph.to_diagrams() integration**

### Phase 3: API and Error Handling
9. **Create FastAPI endpoints for stateless diagram generation**
10. **Add placeholder chat endpoint with NotImplementedError**
11. **Implement comprehensive error handling and validation**
12. **Add request/response schemas and documentation**

### Phase 4: Testing and Validation
13. **Write unit tests for all node functions**
14. **Create integration tests for full workflow**
15. **Add API endpoint tests with sample requests**
16. **Performance testing and optimization**

## Validation Gates

### Code Quality
```bash
# Style and type checking
ruff check --fix diagram_agent/
mypy diagram_agent/

# Security scanning
bandit -r diagram_agent/
```

### Unit Tests
```bash
# Run all tests with coverage
pytest tests/ -v --cov=diagram_agent --cov-report=html

# Specific test categories
pytest tests/unit/ -v  # Unit tests
pytest tests/integration/ -v  # Integration tests
```

### API Testing
```bash
# Start server and test endpoints
uvicorn diagram_agent.api.endpoints:app --host 0.0.0.0 --port 8000 &
curl -X POST "http://localhost:8000/generate-diagram" \
  -H "Content-Type: application/json" \
  -d '{"message": "Create microservices architecture"}'
```

### Integration Validation  
```bash
# Test MCP server connectivity
cd microservices/mcp_server && python mcp_graph_server.py &
cd microservices/diagram_agent && python -c "
from diagram_agent.tools.mcp_client import setup_mcp_client
client = setup_mcp_client()
tools = await client.get_tools()
print(f'Available tools: {[t.name for t in tools]}')
"
```

## Dependencies and Environment

### Core Dependencies (from pyproject.toml)
```toml
dependencies = [
    "langgraph>=0.2.0",          # Core LangGraph functionality
    "langchain-openai>=0.1.0",   # LLM integration  
    "langchain-mcp-adapters",     # MCP client integration
    "fastapi>=0.104.1",          # API framework
    "pydantic>=2.5.0",           # Data validation
    "diagrams>=0.24.4",          # Final diagram output
    "mcp[cli]>=1.0.0",           # MCP protocol support
]
```

### Environment Setup
```bash
# Activate virtual environment
source venv_linux/bin/activate

# Install dependencies
pip install -e .

# Set required environment variables
export OPENAI_API_KEY="your-api-key-here"
export MODEL_NAME="gpt-4o-mini"
```

## Risk Assessment and Mitigation

### Technical Risks
1. **MCP Integration Complexity**: Mitigated by using established langchain-mcp-adapters
2. **State Management**: Use proven TypedDict patterns with proper reducers
3. **Error Propagation**: Comprehensive error handling at each node
4. **LLM Reliability**: Implement retry logic and fallback strategies

### Implementation Risks  
1. **Over-Engineering**: Focus on stateless mode first, placeholder for chat
2. **Tool Planning Complexity**: Start with simple heuristic-based planning
3. **Performance**: Async implementation throughout the pipeline

## Success Criteria

### Functional Requirements
- ✅ Agent successfully parses user diagram requests
- ✅ Tool planning produces executable MCP tool sequences  
- ✅ MCP integration calls all graph server tools correctly
- ✅ Graph.to_diagrams() conversion produces valid Diagram objects
- ✅ FastAPI endpoints handle requests and return proper responses
- ✅ Error handling gracefully manages failures at each step

### Quality Requirements
- ✅ All unit tests pass with >90% coverage
- ✅ Integration tests validate end-to-end workflows
- ✅ API tests confirm proper request/response handling
- ✅ Code passes linting (ruff) and type checking (mypy)
- ✅ Performance: <30s response time for typical diagram generation

### Documentation Requirements
- ✅ API documentation with example requests/responses
- ✅ Architecture documentation explaining node functions
- ✅ Deployment guide with environment setup
- ✅ Usage examples with common diagram types

## Confidence Score: 8.5/10

**Rationale**: High confidence due to:
- ✅ **Strong Foundation**: Existing MCP server and graph structure provide solid base
- ✅ **Proven Patterns**: Modern LangGraph patterns are well-documented and tested
- ✅ **Clear Requirements**: Well-defined 4-node architecture with specific responsibilities  
- ✅ **Incremental Approach**: Stateless-first implementation reduces complexity
- ✅ **Comprehensive Testing**: Detailed testing strategy covers all integration points

**Risk Factors**: 
- ⚠️ **LLM Planning Reliability**: Tool sequence planning may require iteration
- ⚠️ **MCP Transport**: stdio transport reliability in production environment
- ⚠️ **Diagram Conversion**: Potential edge cases in Graph.to_diagrams() integration

This PRP provides comprehensive context and clear implementation path for successful one-pass development of the LangGraph diagram agent.