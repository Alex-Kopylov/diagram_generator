"""
LangGraph-based diagram generation agent.

This agent uses native tools for diagram generation, implementing a workflow
with separate planner and executor nodes orchestrated by a StateGraph.
"""

import base64
import time
from collections.abc import Sequence
from typing import Annotated, Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agents.prompts import DIAGRAM_EXECUTOR_PROMPT, DIAGRAM_PLANNER_PROMPT
from config.settings import get_settings
from core.graph_structure import Graph
from tools.graph_tools import (
    EXECUTOR_TOOLS,
    PLANNER_TOOLS,
    generate_diagram,
)

settings = get_settings()


class PlannerState(TypedDict):
    """State for the planner React agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    plan: str | None


class ExecutorState(TypedDict):
    """State for the executor React agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    graph: Graph | None


class DiagramState(BaseModel):
    """
    Represents the state of the diagram generation workflow.
    
    Attributes:
        message: Original user message describing the diagram
        plan: Generated execution plan from the planner
        result: Final result from the executor
        graph: Graph object created by executor
        diagram_result: Final diagram generation result
        success: Whether the workflow completed successfully
        error: Error message if workflow failed
    """
    message: str = Field(..., description="Original user message describing the diagram")
    plan: str | None = Field(None, description="Generated execution plan from the planner")
    result: str | None = Field(None, description="Final result from the executor")
    graph: Graph | None = Field(None, description="Graph object created by executor")
    diagram_result: dict[str, Any] | None = Field(None, description="Final diagram generation result")
    success: bool = Field(..., description="Whether the workflow completed successfully")
    error: str | None = Field(None, description="Error message if workflow failed")


class DiagramGenerationResult(BaseModel):
    """Result of diagram generation process."""
    success: bool = Field(..., description="Whether diagram generation succeeded")
    message: str = Field(..., description="Success or error message")
    bytestring_base64: str | None = Field(None, description="Generated diagram as base64 encoded string")
    graph_data: Graph | None = Field(None, description="Graph data used for diagram")


# React Agent Helper Functions

def create_planner_tools_node():
    """Create tools node for planner with PLANNER_TOOLS."""
    return ToolNode(PLANNER_TOOLS)


def create_executor_tools_node():
    """Create tools node for executor with EXECUTOR_TOOLS."""
    # Create a custom executor tool node that can store graph state
    def executor_tool_node(state: ExecutorState) -> ExecutorState:
        """Execute executor tool calls and store graph if build_graph is called."""
        # Use ToolNode to handle the actual tool execution
        tool_node = ToolNode(EXECUTOR_TOOLS)
        tool_result = tool_node.invoke({"messages": state["messages"]})

        # Start with the tool result and preserve the existing state
        updated_state = {
            "messages": tool_result["messages"],
            "graph": state.get("graph")  # Preserve existing graph
        }

        # Check if any graph-building tool was called and extract the graph
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                graph_building_tools = {"build_graph", "add_to_graph", "add_node_to_graph", "add_edge_to_graph", "create_empty_graph"}
                if tool_call["name"] in graph_building_tools:
                    # Find the corresponding tool result message
                    for tool_msg in tool_result["messages"]:
                        if (hasattr(tool_msg, 'name') and tool_msg.name == tool_call["name"] and
                            hasattr(tool_msg, 'tool_call_id') and tool_msg.tool_call_id == tool_call["id"]):
                            # Extract the actual Graph object from the tool result
                            try:
                                # Re-execute the tool to get the actual Graph object
                                match tool_call["name"]:
                                    case "build_graph":
                                        from tools.graph_tools import build_graph
                                        graph_result = build_graph.invoke(tool_call["args"])
                                    case "add_to_graph":
                                        from tools.graph_tools import add_to_graph
                                        graph_result = add_to_graph.invoke(tool_call["args"])
                                    case "add_node_to_graph":
                                        from tools.graph_tools import add_node_to_graph
                                        graph_result = add_node_to_graph.invoke(tool_call["args"])
                                    case "add_edge_to_graph":
                                        from tools.graph_tools import add_edge_to_graph
                                        graph_result = add_edge_to_graph.invoke(tool_call["args"])
                                    case "create_empty_graph":
                                        from tools.graph_tools import create_empty_graph
                                        graph_result = create_empty_graph.invoke(tool_call["args"])

                                updated_state["graph"] = graph_result
                                logger.info(f"Graph stored in executor state from {tool_call['name']}: {graph_result.name} with {len(graph_result.nodes)} nodes")
                                break
                            except Exception as e:
                                logger.error(f"Failed to extract graph from {tool_call['name']} call: {e}")
                                logger.debug(f"Tool call args: {tool_call['args']}")

        return updated_state

    return executor_tool_node


def should_continue_planner(state: PlannerState):
    """Decide whether to continue or end the planner."""
    last_message = state["messages"][-1]
    return "continue" if last_message.tool_calls else "end"


def should_continue_executor(state: ExecutorState):
    """Decide whether to continue or end the executor."""
    last_message = state["messages"][-1]

    # If no tool calls, we're done
    if not last_message.tool_calls:
        return "end"

    # If build_graph was called, we should stop after executing it
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "build_graph":
            # Check if this is the first time we're seeing build_graph
            # If graph is already built, we should end
            if graph:=state.get("graph"):
                if graph.node_count() > 0:
                    return "end"

    return "continue"


def create_planner_agent_node():
    """Create the planner agent node using React pattern."""
    def planner_agent(state: PlannerState, config: RunnableConfig) -> PlannerState:
        """Planner agent node that calls model with tools."""
        # Initialize model with tools
        llm = init_chat_model(model=settings.model_name, temperature=settings.temperature, api_key=settings.model_api_key).with_fallbacks([
            init_chat_model(model=settings.fallback_model_name, temperature=settings.temperature, api_key=settings.fallback_model_api_key)
        ])
        llm = llm.bind_tools(PLANNER_TOOLS)

        # Invoke model with current messages
        response = llm.invoke(state["messages"], config)
        return {"messages": [response]}

    return planner_agent


def create_executor_agent_node():
    """Create the executor agent node using React pattern."""
    def executor_agent(state: ExecutorState, config: RunnableConfig) -> ExecutorState:
        """Executor agent node that calls model with tools."""
        # Initialize model with tools
        llm = init_chat_model(model=settings.model_name, temperature=settings.temperature, api_key=settings.model_api_key).with_fallbacks([
                    init_chat_model(model=settings.fallback_model_name, temperature=settings.temperature, api_key=settings.fallback_model_api_key)
                ])
        llm = llm.bind_tools(EXECUTOR_TOOLS)

        # Invoke model with current messages
        response = llm.invoke(state["messages"], config)
        return {"messages": [response]}

    return executor_agent


# Node Functions for LangGraph Workflow

def planner_node(state: DiagramState) -> DiagramState:
    """
    Planner node that creates a detailed execution plan using React agent pattern.
    
    Args:
        state: Current diagram workflow state
        
    Returns:
        Updated state with generated plan
    """
    logger.info("---PLANNER NODE---")
    try:
        # Create planner React agent
        planner_workflow = StateGraph(PlannerState)

        # Add nodes
        planner_agent_node = create_planner_agent_node()
        planner_tools_node = create_planner_tools_node()

        planner_workflow.add_node("agent", planner_agent_node)
        planner_workflow.add_node("tools", planner_tools_node)

        # Set entry point
        planner_workflow.set_entry_point("agent")

        # Add conditional edges
        planner_workflow.add_conditional_edges(
            "agent",
            should_continue_planner,
            {
                "continue": "tools",
                "end": END
            }
        )
        planner_workflow.add_edge("tools", "agent")

        # Compile the planner agent
        planner_agent = planner_workflow.compile()

        # Prepare initial state
        prompt_content = f"Create a detailed plan for: {state.message}"
        logger.info(f"LLM Call - Model: {settings.model_name}, Temperature: {settings.temperature}, Agent: planner, Prompt: {prompt_content}")

        initial_state = PlannerState(
            messages=[
                SystemMessage(content=DIAGRAM_PLANNER_PROMPT),
                HumanMessage(content=prompt_content)
            ],
            plan=None
        )

        # Execute the planner agent
        start_time = time.time()
        final_state = planner_agent.invoke(initial_state, {"recursion_limit": 70})

        # Extract plan from final messages
        plan = "No plan generated"
        if final_state["messages"]:
            # Find the last AI message that's not a tool call
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    plan = msg.content
                    break

        logger.info(f"Planner completed with plan: {plan[:100]}...")

        state.plan = plan
        state.success = True
        return state

    except Exception as e:
        logger.exception(f"LLM Error - Model: {settings.model_name}, Agent: planner, Error: {str(e)}, Prompt: Create a detailed plan for: {state.message}")
        logger.error(f"Planning failed: {str(e)}")
        state.plan = None
        state.success = False
        state.error = f"Planning failed: {str(e)}"
        return state


def executor_node(state: DiagramState) -> DiagramState:
    """
    Executor node that executes the generated plan using React agent pattern.
    
    Args:
        state: Current diagram workflow state with plan
        
    Returns:
        Updated state with execution result and built graph
    """
    logger.info("---EXECUTOR NODE---")
    try:
        if not state.plan:
            logger.error("No plan available for execution")
            state.success = False
            state.error = "No plan available for execution"
            return state

        # Create executor React agent
        executor_workflow = StateGraph(ExecutorState)

        # Add nodes
        executor_agent_node = create_executor_agent_node()
        executor_tools_node = create_executor_tools_node()

        executor_workflow.add_node("agent", executor_agent_node)
        executor_workflow.add_node("tools", executor_tools_node)

        # Set entry point
        executor_workflow.set_entry_point("agent")

        # Add conditional edges
        executor_workflow.add_conditional_edges(
            "agent",
            should_continue_executor,
            {
                "continue": "tools",
                "end": END
            }
        )
        executor_workflow.add_edge("tools", "agent")

        executor_agent = executor_workflow.compile()

        # Prepare initial state
        prompt_content = f"Execute this plan and MUST end with build_graph: {state.plan}"
        logger.info(f"LLM Call - Model: {settings.model_name}, Temperature: {settings.temperature}, Agent: executor, Prompt: {prompt_content}")

        initial_state = ExecutorState(
            messages=[
                SystemMessage(content=DIAGRAM_EXECUTOR_PROMPT),
                HumanMessage(content=prompt_content)
            ],
            graph=None
        )

        # Execute the executor agent
        start_time = time.time()
        final_state = executor_agent.invoke(initial_state, {"recursion_limit": 50})

        # Extract result and graph from final state
        result_message = "Execution completed"
        if final_state["messages"]:
            # Find the last AI message
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    result_message = msg.content
                    break

        # Get the built graph from final state
        built_graph = final_state.get("graph")

        if built_graph:
            state.graph = built_graph
            logger.info(f"Graph successfully built by executor: {built_graph.name} with {len(built_graph.nodes)} nodes")
        else:
            logger.warning("No graph was built during execution")
            # Don't create an empty graph here - let graph_builder_node handle the error

        state.result = result_message
        state.success = True

        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Executor completed plan execution in {duration_ms:.2f}ms")

        return state

    except Exception as e:
        logger.exception(f"LLM Error - Model: {settings.model_name}, Agent: executor, Error: {str(e)}, Prompt: Execute this plan: {state.plan if state.plan else 'No plan'}")
        logger.error(f"Execution failed: {str(e)}")
        state.result = None
        state.success = False
        state.error = f"Execution failed: {str(e)}"
        return state


def graph_builder_node(state: DiagramState) -> DiagramState:
    """
    Graph builder node that generates the final diagram from the built graph.
    Handles all post-processing logic including diagram generation, file output,
    and final result formatting.
    
    Args:
        state: Current diagram workflow state with graph from executor
        
    Returns:
        Updated DiagramState with complete diagram generation results
    """
    logger.info("---GRAPH_BUILDER NODE---")
    try:
        if not state.graph:
            logger.error("No graph available for diagram generation")
            state.success = False
            state.error = "No graph available for diagram generation"
            return state

        # Generate diagram by calling the tool with proper input
        diagram_result = generate_diagram.invoke({
            "graph": state.graph,
        })

        # Update state with complete diagram generation results
        state.diagram_result = diagram_result.model_dump()
        state.success = diagram_result.success

        if not diagram_result.success:
            state.error = diagram_result.error
            logger.error(f"Diagram generation failed: {diagram_result.error}")
        else:
            # Set success message if not already set
            if not state.result:
                state.result = "Diagram generated successfully"
            logger.info("Diagram generated successfully")

        return state

    except Exception as e:
        logger.exception(f"Graph builder failed: {str(e)}")
        state.success = False
        state.error = f"Graph builder failed: {str(e)}"
        return state


class DiagramAgent:
    """
    LangGraph-based diagram generation agent.
    
    This agent orchestrates planner and executor nodes using a StateGraph workflow.
    """

    def __init__(self, model_name: str = settings.model_name, temperature: float = settings.temperature):
        """
        Initialize the diagram generation agent.
        
        Args:
            model_name: OpenAI model to use
            temperature: Model temperature for response generation
        """
        self.model_name = model_name
        self.temperature = temperature

        # Build the LangGraph workflow
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """
        Build the LangGraph workflow with planner and executor nodes.
        
        Returns:
            Compiled StateGraph workflow
        """
        # Create the workflow
        workflow = StateGraph(DiagramState)

        # Add nodes
        workflow.add_node("planner", planner_node)
        workflow.add_node("executor", executor_node)
        workflow.add_node("graph_builder", graph_builder_node)

        # Define edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "graph_builder")
        workflow.add_edge("graph_builder", END)

        # Compile the workflow
        return workflow.compile()

    async def generate_diagram(self, message: str, output_file: str | None = None) -> DiagramGenerationResult:
        """
        Generate a diagram from a natural language description using the workflow.
        Acts as orchestrator - passes args to workflow and returns output from nodes.
        
        Args:
            message: User message describing the diagram to create
            output_file: Optional output file name
            
        Returns:
            DiagramGenerationResult: Result of the diagram generation process
        """
        logger.info(f"Starting diagram generation for message: {message}")
        workflow_start_time = time.time()

        try:
            # Initialize state
            initial_state = DiagramState(
                message=message,
                plan=None,
                result=None,
                graph=None,
                diagram_result=None,
                success=False,
                error=None,
            )

            # Execute the workflow
            final_state = await self.workflow.ainvoke(initial_state)

            workflow_duration_ms = (time.time() - workflow_start_time) * 1000
            logger.info(f"Diagram generation workflow completed in {workflow_duration_ms:.2f}ms")

            # Extract results from final state (post-processing handled by graph_builder_node)
            bytestring_base64_data = None
            if final_state["diagram_result"] and final_state["diagram_result"].get("bytestring"):
                # Convert bytes to base64 string for JSON serialization
                bytestring_data = final_state["diagram_result"]["bytestring"]
                if bytestring_data:
                    bytestring_base64_data = base64.b64encode(bytestring_data).decode('utf-8')

            return DiagramGenerationResult(
                success=final_state["success"],
                message=final_state["result"] or final_state["error"] or "Workflow completed",
                bytestring_base64=bytestring_base64_data,
                graph_data=final_state["graph"]
            )

        except Exception as e:
            workflow_duration_ms = (time.time() - workflow_start_time) * 1000
            logger.exception(f"Diagram generation workflow failed after {workflow_duration_ms:.2f}ms: {str(e)}")

            return DiagramGenerationResult(
                success=False,
                message=f"Diagram generation workflow failed: {str(e)}",
                bytestring_base64=None,
                graph_data=None
            )

    async def chat(self, message: str, session_id: str | None = None) -> str:
        """
        Handle conversational interactions using the workflow.
        
        Args:
            message: User message
            session_id: Optional session ID for conversation tracking
            
        Returns:
            str: Agent response
        """
        logger.info(f"Starting chat interaction with session_id: {session_id}")
        try:
            result = await self.generate_diagram(message)
            logger.info(f"Chat interaction completed successfully for session_id: {session_id}")
            return result.message

        except Exception as e:
            logger.error(f"Chat interaction failed for session_id: {session_id}: {str(e)}")
            return f"Error in chat: {str(e)}"

# Factory functions for easy agent creation
def create_diagram_agent(model_name: str = settings.model_name, temperature: float = settings.temperature) -> DiagramAgent:
    """Create a diagram generation agent with LangGraph workflow."""
    return DiagramAgent(model_name=model_name, temperature=temperature)
