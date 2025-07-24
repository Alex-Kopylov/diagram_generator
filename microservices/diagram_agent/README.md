# LangGraph Diagram Agent

A unified microservice for generating diagrams using native LangGraph tools, eliminating the need for external MCP server communication.

## 🎯 Features

- **Native LangGraph Tools**: Direct integration without MCP server dependency
- **Unified Architecture**: Single service with embedded graph construction tools  
- **FastAPI REST API**: Clean HTTP endpoints for diagram generation
- **Conversational Interface**: Chat-based diagram creation and refinement
- **Docker Support**: Containerized for easy deployment
- **Health Monitoring**: Built-in health checks and observability

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│         Diagram Agent               │
│  ┌─────────────┐ ┌─────────────────┐│
│  │  LangGraph  │─│  Native Tools   ││
│  │   Agent     │ │ (Graph Const.)  ││
│  └─────────────┘ └─────────────────┘│
│ ┌─────────────────────────────────┐ │
│ │        FastAPI Endpoints        │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

## 📦 Components

### Core Components
- **Native Tools**: Direct graph construction without network calls
- **LangGraph Agent**: Intelligent diagram planning and execution  
- **FastAPI Endpoints**: RESTful API for diagram generation
- **Graph Structure**: Embedded graph data models and operations

### Key Improvements
- ✅ **Zero Network Latency**: No MCP server communication
- ✅ **Simplified Deployment**: Single container service
- ✅ **Better Reliability**: No external service dependencies
- ✅ **Enhanced Performance**: Direct function calls vs HTTP requests
- ✅ **Easier Debugging**: All logic in one service

## 🚀 Installation

### Using UV (Recommended)

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
```

### Using Pip

```bash
pip install -e .
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```env
# OpenAI Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Service Configuration
PORT=3502
HOST=0.0.0.0
LOG_LEVEL=info
RELOAD=false

# Optional: Model Configuration
OPENAI_MODEL=gpt-4
TEMPERATURE=0.1
```

## 🎮 Usage

### Starting the Service

#### Development Mode
```bash
# Using the main module
python main.py

# Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 3502 --reload
```

#### Production Mode
```bash
# Using Docker
docker build -t diagram-agent .
docker run -p 3502:3502 diagram-agent
```

### 🔗 API Endpoints

#### Health Check
```bash
curl http://localhost:3502/health
```

Response:
```json
{
  "status": "healthy",  
}
```

#### Generate Diagram
```bash
curl -X POST http://localhost:3502/generate-diagram \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a microservices architecture with API gateway, authentication service, and database",
    "output_file": "microservices.png"
  }'
```

#### Interactive Chat
```bash
curl -X POST http://localhost:3502/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What types of diagrams can you create?",
    "session_id": "user123"
  }'
```

#### Service Information
```bash
curl http://localhost:3502/
```

Response:
```json
{
  "service": "LangGraph Diagram Agent",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "generate_diagram": "/generate-diagram", 
    "chat": "/chat",
    "docs": "/docs"
  },
  "features": [
    "Native LangGraph tools",
    "Conversational diagram generation",
    "Direct graph construction"
  ]
}
```

## 📊 Available Tools

The agent includes these native tools:

1. **create_node**: Create diagram components
2. **create_edge**: Connect components with relationships
3. **create_cluster**: Group related components
4. **build_graph**: Construct complete graphs
5. **add_to_graph**: Extend existing graphs
6. **validate_graph**: Verify graph structure
8. **generate_diagram**: Render final diagrams

## 🎨 Example Use Cases

### Basic Web Application
```json
{
  "message": "Create a web application with load balancer, web servers, and database",
  "output_file": "webapp.png"
}
```

### Microservices Architecture  
```json
{
  "message": "Design a microservices system with API gateway, user service, payment service, and shared database",
  "output_file": "microservices.png"
}
```

### Cloud Infrastructure
```json
{
  "message": "Show AWS infrastructure with VPC, public/private subnets, EC2 instances, and RDS database",
  "output_file": "aws_infra.png"
}
```

### Conversational Refinement
```json
{
  "message": "Add a Redis cache between the API gateway and services",
  "session_id": "user123"
}
```

## 🧪 Development

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=diagram_agent --cov-report=html

# Run integration tests specifically
pytest tests/test_integration.py -v
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code  
ruff check .

# Type checking
mypy diagram_agent/
```

## 🐳 Deployment

### Simplified Docker Deployment

Since the service is now unified, deployment is much simpler:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .

EXPOSE 3502

CMD ["python", "main.py"]
```

### Docker Compose (Optional)

```yaml
version: '3.8'
services:
  diagram-agent:
    build: .
    ports:
      - "3502:3502"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PORT=3502
    volumes:
      - ./output:/app/output
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Verify `OPENAI_API_KEY` is set correctly
   - Check API usage limits and billing
   - Ensure model availability (gpt-4, gpt-3.5-turbo)

2. **Diagram Generation Failures**
   - Check if `diagrams` library is installed correctly
   - Verify Graphviz is installed: `apt-get install graphviz` (Linux) or `brew install graphviz` (Mac)
   - Check file permissions for output directory

3. **Import Errors**
   - Ensure all dependencies are installed: `uv sync`
   - Check Python path includes the project directory

### Debugging

Enable debug logging:
```bash
export LOG_LEVEL=debug
python main.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.