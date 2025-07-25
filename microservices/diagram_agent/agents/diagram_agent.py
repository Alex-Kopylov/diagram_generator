"""
LangGraph-based diagram generation agent.

This agent uses native tools for diagram generation, implementing a workflow
with separate planner and executor nodes orchestrated by a StateGraph.
"""

import time
from typing import Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from pydantic import BaseModel, Field
from loguru import logger

from tools.graph_tools import ALL_GRAPH_TOOLS, PLANNER_TOOLS, EXECUTOR_TOOLS, generate_diagram
from core.graph_structure import Graph
from agents.prompts import DIAGRAM_PLANNER_PROMPT, DIAGRAM_EXECUTOR_PROMPT
from config.settings import get_settings
from langchain.chat_models import init_chat_model

settings = get_settings()


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
        file_path: Path to generated diagram file
        graph_data: Graph data used for diagram generation
    """
    message: str = Field(..., description="Original user message describing the diagram")
    plan: Optional[str] = Field(None, description="Generated execution plan from the planner")
    result: Optional[str] = Field(None, description="Final result from the executor")
    graph: Optional[Graph] = Field(None, description="Graph object created by executor")
    diagram_result: Optional[Dict[str, Any]] = Field(None, description="Final diagram generation result")
    success: bool = Field(..., description="Whether the workflow completed successfully")
    error: Optional[str] = Field(None, description="Error message if workflow failed")
    file_path: Optional[str] = Field(None, description="Path to generated diagram file")
    graph_data: Optional[Dict[str, Any]] = Field(None, description="Graph data used for diagram")


class DiagramGenerationResult(BaseModel):
    """Result of diagram generation process."""
    success: bool = Field(..., description="Whether diagram generation succeeded")
    message: str = Field(..., description="Success or error message")
    file_path: Optional[str] = Field(None, description="Path to generated diagram file")
    graph_data: Optional[Dict[str, Any]] = Field(None, description="Graph data used for diagram")


# Node Functions for LangGraph Workflow

def planner_node(state: DiagramState) -> DiagramState:
    """
    Planner node that creates a detailed execution plan and executes discovery tools.
    
    Args:
        state: Current diagram workflow state
        
    Returns:
        Updated state with generated plan and tool results
    """
    logger.info("---PLANNER NODE---")
    try:
        # Initialize planner agent with discovery tools only
        llm = init_chat_model(model=settings.model_name, temperature=settings.temperature, api_key=settings.openai_api_key)
        llm = llm.bind_tools(PLANNER_TOOLS)

        # Log LLM call
        prompt_content = f"Create a detailed plan for: {state.message}"
        logger.info(f"LLM Call - Model: {settings.model_name}, Temperature: {settings.temperature}, Agent: planner, Prompt: {prompt_content}")
        
        # Create messages for conversation
        messages = [
            SystemMessage(content=DIAGRAM_PLANNER_PROMPT),
            HumanMessage(content=prompt_content)
        ]
        
        # Create plan using the planner agent
        start_time = time.time()
        
        # Execute tool calls in a loop until no more tool calls are needed
        tool_node = ToolNode(PLANNER_TOOLS)
        
        while True:
            # Get LLM response
            result: AIMessage = llm.invoke(messages)
            messages.append(result)
            
            # Check if there are tool calls to execute
            if result.tool_calls:
                logger.info(f"Executing {len(result.tool_calls)} tool calls")
                # Execute tools and get results
                tool_messages = tool_node.invoke({"messages": messages})
                # Add tool results to messages
                messages.extend(tool_messages["messages"])
            else:
                # No more tool calls, we're done
                break
        
        # Log LLM response
        logger.info(f"Planner completed with {len(messages)} messages")
        plan = result.content if result.content else "No plan generated"

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
    Executor node that executes the generated plan and builds the graph.
    
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
        
        # Initialize executor LLM with creation tools only
        llm = init_chat_model(model=settings.model_name, temperature=settings.temperature, api_key=settings.openai_api_key)
        llm = llm.bind_tools(EXECUTOR_TOOLS)
        
        # Log LLM call
        prompt_content = f"Execute this plan and MUST end with build_graph: {state.plan}"
        logger.info(f"LLM Call - Model: {settings.model_name}, Temperature: {settings.temperature}, Agent: executor, Prompt: {prompt_content}")
        
        # Execute the plan using tool execution loop
        start_time = time.time()
        
        messages = [
            SystemMessage(content=DIAGRAM_EXECUTOR_PROMPT),
            HumanMessage(content=prompt_content)
        ]
        
        # Create tool node for executor tools
        tool_node = ToolNode(EXECUTOR_TOOLS)
        built_graph = None
        
        # Execute tool calls in a loop until completion
        while True:
            # Get LLM response
            result: AIMessage = llm.invoke(messages)
            messages.append(result)
            
            # Check if there are tool calls to execute
            if result.tool_calls:
                logger.info(f"Executing {len(result.tool_calls)} tool calls")
                # Execute tools and get results
                tool_result = tool_node.invoke({"messages": messages})
                # Add tool results to messages  
                messages.extend(tool_result["messages"])
                
                # Check if build_graph was called by looking at tool calls and responses
                for i, tool_call in enumerate(result.tool_calls):
                    if tool_call['name'] == 'build_graph':
                        # Find corresponding tool message in responses
                        if i < len(tool_result["messages"]):
                            tool_msg = tool_result["messages"][i]
                            if hasattr(tool_msg, 'content'):
                                # The build_graph tool returns a Graph object as content
                                try:
                                    # The content should be the Graph object directly
                                    built_graph = tool_msg.content
                                    logger.info("Graph successfully built by executor")
                                    break
                                except Exception as graph_error:
                                    logger.error(f"Failed to extract graph: {graph_error}")
                
                # If we found a built graph, we can stop
                if built_graph:
                    break
            else:
                # No more tool calls, we're done
                break
        
        # Set the graph in state
        if built_graph:
            state.graph = built_graph
        else:
            logger.warning("No graph was built during execution, creating empty graph")
            from core.graph_structure import Graph, Direction
            state.graph = Graph(name="Generated Diagram", direction=Direction.LEFT_RIGHT)
        
        state.result = result.content if result.content else "Execution completed"
        state.success = True
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Executor completed plan execution in {duration_ms:.2f}ms")
        
        return state
        
    except Exception as e:
        logger.exception(f"LLM Error - Model: {settings.model_name}, Agent: executor, Error: {str(e)}, Prompt: Execute this plan: {state.get('plan', 'No plan')}")
        logger.error(f"Execution failed: {str(e)}")
        state.result = None
        state.success = False
        state.error = f"Execution failed: {str(e)}"
        return state


def graph_builder_node(state: DiagramState) -> DiagramState:
    """
    Graph builder node that generates the final diagram from the built graph.
    
    Args:
        state: Current diagram workflow state with graph from executor
        
    Returns:
        Updated state with diagram generation result
    """
    logger.info("---GRAPH_BUILDER NODE---")
    try:
        if not state.graph:
            logger.error("No graph available for diagram generation")
            state.success = False
            state.error = "No graph available for diagram generation"
            return state
        
        # Generate diagram by calling the tool with proper input
        from tools.graph_tools import GenerateDiagramInput
        diagram_input = GenerateDiagramInput(
            graph=state.graph,
            output_file=state.file_path
        )
        diagram_result = generate_diagram.invoke(diagram_input.model_dump())
        
        # Update state with diagram result
        state.diagram_result = diagram_result.model_dump()
        state.file_path = diagram_result.file_path
        state.success = diagram_result.success
        
        if not diagram_result.success:
            state.error = diagram_result.error
            logger.error(f"Diagram generation failed: {diagram_result.error}")
        else:
            logger.info(f"Diagram generated successfully: {diagram_result.file_path}")
        
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
    
    async def generate_diagram(self, message: str, output_file: Optional[str] = None) -> DiagramGenerationResult:
        """
        Generate a diagram from a natural language description using the workflow.
        
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
                file_path=output_file,
                graph_data=None
            )
            
            # Execute the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            workflow_duration_ms = (time.time() - workflow_start_time) * 1000
            logger.info(f"Diagram generation workflow completed in {workflow_duration_ms:.2f}ms")
            
            result = DiagramGenerationResult(
                success=final_state.get("success", False),
                message=final_state.get("result") or final_state.get("error", "Workflow completed"),
                file_path=final_state.get("file_path"),
                graph_data=final_state.get("graph_data")
            )
            
            if result.success:
                logger.info(f"Diagram generation successful: {result.file_path}")
            else:
                logger.error(f"Diagram generation failed: {result.message}")
            
            return result
            
        except Exception as e:
            workflow_duration_ms = (time.time() - workflow_start_time) * 1000
            logger.exception(f"Diagram generation workflow failed after {workflow_duration_ms:.2f}ms: {str(e)}")
            
            return DiagramGenerationResult(
                success=False,
                message=f"Diagram generation workflow failed: {str(e)}",
                file_path=None,
                graph_data=None
            )
    
    async def chat(self, message: str, session_id: Optional[str] = None) -> str:
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