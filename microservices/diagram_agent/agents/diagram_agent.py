"""
LangGraph-based diagram generation agent.

This agent uses native tools for diagram generation, implementing a workflow
with separate planner and executor nodes orchestrated by a StateGraph.
"""

import time
from typing import Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from loguru import logger

from tools.graph_tools import ALL_GRAPH_TOOLS
from agents.prompts import DIAGRAM_PLANNER_PROMPT, DIAGRAM_EXECUTOR_PROMPT
from config.settings import get_settings
from langchain.chat_models import init_chat_model

settings = get_settings()


class DiagramState(TypedDict):
    """
    Represents the state of the diagram generation workflow.
    
    Attributes:
        message: Original user message describing the diagram
        plan: Generated execution plan from the planner
        result: Final result from the executor
        success: Whether the workflow completed successfully
        error: Error message if workflow failed
        file_path: Path to generated diagram file
        graph_data: Graph data used for diagram generation
    """
    message: str
    plan: Optional[str]
    result: Optional[str]
    success: bool
    error: Optional[str]
    file_path: Optional[str]
    graph_data: Optional[Dict[str, Any]]


class DiagramGenerationResult(BaseModel):
    """Result of diagram generation process."""
    success: bool = Field(..., description="Whether diagram generation succeeded")
    message: str = Field(..., description="Success or error message")
    file_path: Optional[str] = Field(None, description="Path to generated diagram file")
    graph_data: Optional[Dict[str, Any]] = Field(None, description="Graph data used for diagram")


# Node Functions for LangGraph Workflow

def planner_node(state: DiagramState) -> DiagramState:
    """
    Planner node that creates a detailed execution plan.
    
    Args:
        state: Current diagram workflow state
        
    Returns:
        Updated state with generated plan
    """
    logger.info("---PLANNER NODE---")
    try:
        # Initialize planner agent
        llm = init_chat_model(model=settings.model_name, temperature=settings.temperature, api_key=settings.openai_api_key)

        # Log LLM call
        prompt_content = f"Create a detailed plan for: {state['message']}"
        logger.info(f"LLM Call - Model: {settings.model_name}, Temperature: {settings.temperature}, Agent: planner, Prompt: {prompt_content}")
        
        # Create plan using the planner agent
        start_time = time.time()
        result = llm.invoke([
                SystemMessage(content=DIAGRAM_PLANNER_PROMPT),
                HumanMessage(content=prompt_content)
            ],
        )
        duration_ms = (time.time() - start_time) * 1000
        
        # Log LLM response
        logger.info(f"LLM Response - Model: {settings.model_name}, Agent: planner, Duration: {duration_ms:.2f}ms, Response: {str(result)}")
        plan = result["messages"][-1].content if result["messages"] else "No plan generated"

        
        return {
            **state,
            "plan": plan,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"LLM Error - Model: {settings.model_name}, Agent: planner, Error: {str(e)}, Prompt: Create a detailed plan for: {state['message']}")
        logger.error(f"Planning failed: {str(e)}")
        return {
            **state,
            "plan": None,
            "success": False,
            "error": f"Planning failed: {str(e)}"
        }


def executor_node(state: DiagramState) -> DiagramState:
    """
    Executor node that executes the generated plan.
    
    Args:
        state: Current diagram workflow state with plan
        
    Returns:
        Updated state with execution result
    """
    logger.info("---EXECUTOR NODE---")
    try:
        if not state.get("plan"):
            logger.error("No plan available for execution")
            return {
                **state,
                "success": False,
                "error": "No plan available for execution"
            }
        
        # Initialize executor agent
        llm = init_chat_model(model=settings.model_name, temperature=settings.temperature, api_key=settings.openai_api_key)
        
        executor_agent = create_react_agent(
            model=llm,
            tools=ALL_GRAPH_TOOLS,
            prompt=DIAGRAM_EXECUTOR_PROMPT
        )
        
        # Log LLM call
        prompt_content = f"Execute this plan: {state['plan']}"
        logger.info(f"LLM Call - Model: {settings.model_name}, Temperature: {settings.temperature}, Agent: executor, Prompt: {prompt_content}")
        
        # Execute the plan
        start_time = time.time()
        result = executor_agent.invoke([
            SystemMessage(content=DIAGRAM_EXECUTOR_PROMPT),
            AIMessage(content=f"Plan: {state['plan']}"),
            HumanMessage(content=prompt_content)],
        )
        duration_ms = (time.time() - start_time) * 1000
        
        
        # Log LLM response
        logger.info(f"LLM Response - Model: {settings.model_name}, Agent: executor, Duration: {duration_ms:.2f}ms, Response: {str(result)}")
        execution_result = result["messages"][-1].content if result["messages"] else "No execution result"

        logger.info(f"Executor completed plan execution successfully in {duration_ms:.2f}ms")
        
        return {
            **state,
            "result": execution_result,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"LLM Error - Model: {settings.model_name}, Agent: executor, Error: {str(e)}, Prompt: Execute this plan: {state.get('plan', 'No plan')}")
        logger.error(f"Execution failed: {str(e)}")
        return {
            **state,
            "result": None,
            "success": False,
            "error": f"Execution failed: {str(e)}"
        }


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
        
        # Define edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", END)
        
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
            logger.error(f"Diagram generation workflow failed after {workflow_duration_ms:.2f}ms: {str(e)}")
            
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


# Legacy agent classes for backward compatibility (now implemented as nodes)

class PlannerAgent:
    """
    Legacy planner agent - now implemented as a node in the workflow.
    Kept for backward compatibility.
    """
    
    def __init__(self, model_name: str = settings.model_name, temperature: float = settings.temperature):
        """Initialize the planner agent."""
        self.model_name = model_name
        self.temperature = temperature
    
    async def create_plan(self, description: str) -> str:
        """
        Create a detailed execution plan for diagram generation.
        
        Args:
            description: Natural language description of the diagram
            
        Returns:
            str: Detailed execution plan
        """
        # Use the planner node function directly
        state = DiagramState(
            message=description,
            plan=None,
            result=None,
            success=False,
            error=None,
            file_path=None,
            graph_data=None
        )
        
        result_state = planner_node(state)
        return result_state.get("plan") or "No plan generated"


class ExecutorAgent:
    """
    Legacy executor agent - now implemented as a node in the workflow.
    Kept for backward compatibility.
    """
    
    def __init__(self, model_name: str = settings.model_name, temperature: float = settings.temperature):
        """Initialize the executor agent."""
        self.model_name = model_name
        self.temperature = temperature
    
    async def execute_plan(self, plan: str) -> DiagramGenerationResult:
        """
        Execute a diagram generation plan.
        
        Args:
            plan: Detailed execution plan
            
        Returns:
            DiagramGenerationResult: Result of the execution
        """
        # Use the executor node function directly
        state = DiagramState(
            message="",
            plan=plan,
            result=None,
            success=False,
            error=None,
            file_path=None,
            graph_data=None
        )
        
        result_state = executor_node(state)
        
        return DiagramGenerationResult(
            success=result_state.get("success", False),
            message=result_state.get("result") or result_state.get("error") or "Execution completed",
            file_path=result_state.get("file_path"),
            graph_data=result_state.get("graph_data")
        )


# Factory functions for easy agent creation
def create_diagram_agent(model_name: str = settings.model_name, temperature: float = settings.temperature) -> DiagramAgent:
    """Create a diagram generation agent with LangGraph workflow."""
    return DiagramAgent(model_name=model_name, temperature=temperature)


def create_planner_agent(model_name: str = settings.model_name, temperature: float = settings.temperature) -> PlannerAgent:
    """Create a planning agent (legacy compatibility)."""
    return PlannerAgent(model_name=model_name, temperature=temperature)


def create_executor_agent(model_name: str = settings.model_name, temperature: float = settings.temperature) -> ExecutorAgent:
    """Create an executor agent (legacy compatibility)."""
    return ExecutorAgent(model_name=model_name, temperature=temperature)