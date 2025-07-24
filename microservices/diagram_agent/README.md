# Diagram Agent

LangGraph-based intelligent agent for diagram generation using MCP graph server.

## Installation

```bash
cd microservices/diagram_agent
pip install -e .
```

## Usage

Start the FastAPI server:

```bash
uvicorn diagram_agent.api.endpoints:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /generate-diagram` - Stateless diagram generation
- `POST /chat` - Chat endpoint (placeholder - not implemented yet)

## Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
export MODEL_NAME="gpt-4o-mini"
```