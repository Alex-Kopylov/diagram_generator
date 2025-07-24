"""
Integration tests for the unified diagram agent microservice.

Tests the native tools integration and API endpoints.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from ..tools.graph_tools import (
    create_node, create_edge, create_cluster, build_graph,
    validate_graph, generate_diagram
)
from ..agents.diagram_agent import DiagramAgent, create_diagram_agent


class TestNativeTools:
    """Test the native LangGraph tools."""
    
    def test_create_node(self):
        """Test node creation."""
        node = create_node.invoke({"name": "diagrams.aws.compute.EC2", "id": "web-server"})
        
        assert node.name == "diagrams.aws.compute.EC2"
        assert node.id == "web-server"
    
    def test_create_node_auto_id(self):
        """Test node creation with auto-generated ID."""
        node = create_node.invoke({"name": "diagrams.aws.database.RDS"})
        
        assert node.name == "diagrams.aws.database.RDS"
        assert node.id is not None
        assert len(node.id) > 0
    
    def test_create_edge(self):
        """Test edge creation."""
        edge_data = create_edge.invoke({
            "source_id": "web-server",
            "target_id": "database",
            "forward": True,
            "reverse": False
        })
        
        assert edge_data["source_id"] == "web-server"
        assert edge_data["target_id"] == "database"
        assert edge_data["forward"] is True
        assert edge_data["reverse"] is False
        assert "id" in edge_data
    
    def test_create_cluster(self):
        """Test cluster creation."""
        cluster_data = create_cluster.invoke({
            "name": "Web Tier",
            "node_ids": ["web-server-1", "web-server-2"]
        })
        
        assert cluster_data["name"] == "Web Tier"
        assert cluster_data["node_ids"] == ["web-server-1", "web-server-2"]
        assert "id" in cluster_data
    
    def test_build_graph(self):
        """Test complete graph building."""
        nodes = [
            {"name": "diagrams.aws.compute.EC2", "id": "web-server"},
            {"name": "diagrams.aws.database.RDS", "id": "database"}
        ]
        edges = [
            {"source_id": "web-server", "target_id": "database", "forward": True}
        ]
        
        graph = build_graph.invoke({
            "name": "Simple Web App",
            "direction": "LR",
            "nodes": nodes,
            "edges": edges
        })
        
        assert graph.name == "Simple Web App"
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.direction.value == "LR"
    
    def test_validate_graph(self):
        """Test graph validation."""
        # Create a valid graph
        nodes = [
            {"name": "diagrams.aws.compute.EC2", "id": "web-server"},
            {"name": "diagrams.aws.database.RDS", "id": "database"}
        ]
        edges = [
            {"source_id": "web-server", "target_id": "database", "forward": True}
        ]
        
        graph = build_graph.invoke({
            "name": "Test Graph",
            "direction": "TB",
            "nodes": nodes,
            "edges": edges
        })
        
        # Convert to dict for validation tool
        graph_dict = graph.model_dump()
        result = validate_graph.invoke({"graph_data": graph_dict})
        
        assert result.valid is True
        assert len(result.errors) == 0


class TestDiagramAgent:
    """Test the diagram agent integration."""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI API calls."""
        with patch('langchain_openai.ChatOpenAI') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_agent(self):
        """Mock LangGraph agent."""
        with patch('langgraph.prebuilt.create_react_agent') as mock:
            mock_instance = MagicMock()
            mock_instance.ainvoke = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance
    
    @pytest.mark.asyncio
    async def test_agent_creation(self, mock_openai, mock_agent):
        """Test agent creation."""
        agent = create_diagram_agent()
        assert agent is not None
        assert hasattr(agent, 'generate_diagram')
        assert hasattr(agent, 'chat')
    
    @pytest.mark.asyncio
    async def test_generate_diagram_success(self, mock_openai, mock_agent):
        """Test successful diagram generation."""
        # Mock successful agent response
        mock_agent.ainvoke.return_value = {
            "messages": [MagicMock(content="Diagram generated successfully")]
        }
        
        agent = DiagramAgent()
        agent.agent = mock_agent
        
        result = await agent.generate_diagram(
            "Create a simple web application with load balancer and database"
        )
        
        assert result.success is True
        assert "successfully" in result.message
    
    @pytest.mark.asyncio  
    async def test_generate_diagram_error(self, mock_openai, mock_agent):
        """Test diagram generation error handling."""
        # Mock agent error
        mock_agent.ainvoke.side_effect = Exception("Test error")
        
        agent = DiagramAgent()
        agent.agent = mock_agent
        
        result = await agent.generate_diagram("Invalid request")
        
        assert result.success is False
        assert "failed" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_chat_functionality(self, mock_openai, mock_agent):
        """Test chat functionality."""
        # Mock successful chat response
        mock_agent.ainvoke.return_value = {
            "messages": [MagicMock(content="I can help you create diagrams")]
        }
        
        agent = DiagramAgent()
        agent.agent = mock_agent
        
        response = await agent.chat("What can you help me with?")
        
        assert isinstance(response, str)
        assert len(response) > 0


class TestAPIIntegration:
    """Test API endpoint integration."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from ..api.endpoints import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "tools_available" in data
        assert data["tools_available"] > 0
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "LangGraph Diagram Agent"
        assert "endpoints" in data
        assert "features" in data
        assert "Native LangGraph tools" in data["features"]
    
    @patch('diagram_agent.agents.diagram_agent.DiagramAgent.generate_diagram')
    def test_generate_diagram_endpoint(self, mock_generate, client):
        """Test diagram generation endpoint."""
        # Mock successful generation
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.message = "Diagram created successfully"
        mock_result.file_path = "test_diagram.png"
        mock_result.graph_data = None
        mock_generate.return_value = mock_result
        
        response = client.post("/generate-diagram", json={
            "message": "Create a simple web application",
            "output_file": "test.png"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Diagram created successfully"
        assert data["file_path"] == "test_diagram.png"
    
    @patch('diagram_agent.agents.diagram_agent.DiagramAgent.chat')
    def test_chat_endpoint(self, mock_chat, client):
        """Test chat endpoint."""
        # Mock successful chat
        mock_chat.return_value = "I can help you create system diagrams"
        
        response = client.post("/chat", json={
            "message": "What can you do?",
            "session_id": "test-session"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "I can help you create system diagrams"
        assert data["session_id"] == "test-session"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])