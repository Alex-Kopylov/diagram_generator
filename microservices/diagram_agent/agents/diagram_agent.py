"""
LangGraph-based diagram generation agent.

This agent uses native tools for diagram generation, eliminating the need
"""

from typing import Dict, Any, List, Optional, Annotated
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from ..tools.graph_tools import ALL_GRAPH_TOOLS
from .prompts import DIAGRAM_PLANNER_PROMPT, DIAGRAM_EXECUTOR_PROMPT, ASSISTANT_PROMPT
from ..config.settings import get_settings

settings = get_settings()


class DiagramGenerationResult(BaseModel):
    """Result of diagram generation process."""
    success: bool = Field(..., description="Whether diagram generation succeeded")
    message: str = Field(..., description="Success or error message")
    file_path: Optional[str] = Field(None, description="Path to generated diagram file")
    graph_data: Optional[Dict[str, Any]] = Field(None, description="Graph data used for diagram")


class DiagramAgent:
    """
    LangGraph-based diagram generation agent.
    
    This agent processes natural language requests to create system diagrams
    using native graph construction tools.
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.1):
        """
        Initialize the diagram generation agent.
        
        Args:
            model_name: OpenAI model to use
            temperature: Model temperature for response generation
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.openai_api_key
        )
        
        # Create the ReAct agent with graph tools
        self.agent = create_react_agent(
            model=self.llm,
            tools=ALL_GRAPH_TOOLS,
            state_modifier=ASSISTANT_PROMPT
        )
    
    async def generate_diagram(self, message: str, output_file: Optional[str] = None) -> DiagramGenerationResult:
        """
        Generate a diagram from a natural language description.
        
        Args:
            message: User message describing the diagram to create
            output_file: Optional output file name
            
        Returns:
            DiagramGenerationResult: Result of the diagram generation process
        """
        try:
            # Prepare the input with diagram generation context
            enhanced_message = f"""
            Please create a diagram based on this description: {message}
            
            Instructions:
            1. Analyze the requirements and identify all components
            2. Create appropriate nodes using create_node with proper diagrams module paths
            3. Group related components using create_cluster when logical
            4. Connect components using create_edge with appropriate directions
            5. Build the complete graph using build_graph
            6. Generate the final diagram using generate_diagram
            
            Output file: {output_file or 'auto-generated name'}
            
            Make sure to use proper diagrams module paths like:
            - diagrams.aws.compute.EC2
            - diagrams.aws.network.ALB
            - diagrams.aws.database.RDS
            - diagrams.gcp.compute.ComputeEngine
            - diagrams.azure.compute.VirtualMachines
            """
            
            # Execute the agent
            result = await self.agent.ainvoke({
                "messages": [HumanMessage(content=enhanced_message)]
            })
            
            # Extract the final message
            final_message = result["messages"][-1].content if result["messages"] else "No response generated"
            
            return DiagramGenerationResult(
                success=True,
                message=final_message,
                file_path=output_file,
                graph_data=None  # Could be extracted from agent state if needed
            )
            
        except Exception as e:
            return DiagramGenerationResult(
                success=False,
                message=f"Diagram generation failed: {str(e)}",
                file_path=None,
                graph_data=None
            )
    
    async def chat(self, message: str, session_id: Optional[str] = None) -> str:
        """
        Handle conversational interactions about diagrams.
        
        Args:
            message: User message
            session_id: Optional session ID for conversation tracking
            
        Returns:
            str: Agent response
        """
        try:
            result = await self.agent.ainvoke({
                "messages": [HumanMessage(content=message)]
            })
            
            return result["messages"][-1].content if result["messages"] else "No response generated"
            
        except Exception as e:
            return f"Error in chat: {str(e)}"


class PlannerAgent:
    """
    Specialized agent for planning diagram generation.
    
    This agent focuses on analyzing requirements and creating execution plans.
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.1):
        """Initialize the planner agent."""
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.openai_api_key
        )
        
        self.agent = create_react_agent(
            model=self.llm,
            tools=ALL_GRAPH_TOOLS,
            state_modifier=DIAGRAM_PLANNER_PROMPT
        )
    
    async def create_plan(self, description: str) -> str:
        """
        Create a detailed execution plan for diagram generation.
        
        Args:
            description: Natural language description of the diagram
            
        Returns:
            str: Detailed execution plan
        """
        try:
            result = await self.agent.ainvoke({
                "messages": [HumanMessage(content=f"Create a detailed plan for: {description}")]
            })
            
            return result["messages"][-1].content if result["messages"] else "No plan generated"
            
        except Exception as e:
            return f"Planning failed: {str(e)}"


class ExecutorAgent:
    """
    Specialized agent for executing diagram generation plans.
    
    This agent focuses on executing pre-made plans step by step.
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.1):
        """Initialize the executor agent."""
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.openai_api_key
        )
        
        self.agent = create_react_agent(
            model=self.llm,
            tools=ALL_GRAPH_TOOLS,
            state_modifier=DIAGRAM_EXECUTOR_PROMPT
        )
    
    async def execute_plan(self, plan: str) -> DiagramGenerationResult:
        """
        Execute a diagram generation plan.
        
        Args:
            plan: Detailed execution plan
            
        Returns:
            DiagramGenerationResult: Result of the execution
        """
        try:
            result = await self.agent.ainvoke({
                "messages": [HumanMessage(content=f"Execute this plan: {plan}")]
            })
            
            final_message = result["messages"][-1].content if result["messages"] else "No response generated"
            
            return DiagramGenerationResult(
                success=True,
                message=final_message,
                file_path=None,  # Would be determined during execution
                graph_data=None
            )
            
        except Exception as e:
            return DiagramGenerationResult(
                success=False,
                message=f"Execution failed: {str(e)}",
                file_path=None,
                graph_data=None
            )


# Factory functions for easy agent creation
def create_diagram_agent(model_name: str = "gpt-4", temperature: float = 0.1) -> DiagramAgent:
    """Create a diagram generation agent."""
    return DiagramAgent(model_name=model_name, temperature=temperature)


def create_planner_agent(model_name: str = "gpt-4", temperature: float = 0.1) -> PlannerAgent:
    """Create a planning agent."""
    return PlannerAgent(model_name=model_name, temperature=temperature)


def create_executor_agent(model_name: str = "gpt-4", temperature: float = 0.1) -> ExecutorAgent:
    """Create an executor agent."""
    return ExecutorAgent(model_name=model_name, temperature=temperature)