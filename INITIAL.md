## FEATURE:

RESTful Diagram Generation Service

- FastAPI service exposing two endpoints:
  - `POST /diagram` – Generates and returns a diagram (PNG/SVG) from a natural-language description using an agent with diagram creation tools.
  - `POST /assistant` (optional, bonus) – Chat-style endpoint that interprets intent, answers questions, creates diagrams through tool execution, explaining the plan and asking questions back to understand better.

## AGENT ARCHITECTURE:

The service uses an intelligent agent that plans and executes diagram creation through a set of specialized tools. The agent:

1. **Plans** the diagram structure based on the natural language description
2. **Executes** the plan using available tools in a structured sequence
3. **Returns** the generated diagram without exposing underlying Python code

### Available Tools:

1. **Create Diagram** - Initializes a new diagram with specified name, output format, and options
2. **Create Node** - Creates individual nodes (AWS, GCP, Azure, etc.) with specified labels
3. **Create Cluster** - Groups nodes into logical clusters with labels
4. **Create Edge** - Connects nodes with optional labels, colors, and styles
5. **Search Node** - Searches for available node types across cloud providers and services

## EXAMPLES:

See the `/examples` directory for complete working examples of diagram generation scripts:
- `examples/basic_web_application.py` - Basic web application with Load Balancer, EC2 instances, and RDS
- `examples/microservices_architecture.py` - Microservices with API Gateway, SQS, and monitoring

### /diagram – Example 1: Basic Web Application

Request (JSON)
```json
{
  "description": "Create a diagram showing a basic web application with an Application Load Balancer, two EC2 instances for the web servers, and an RDS database for storage. The web servers should be in a cluster named 'Web Tier'."
}
```

Response – `200 OK` with `Content-Type: image/png` containing the rendered diagram.<br/>
Header `X-Diagram-Plan` includes the agent's execution plan.

**Agent Execution Plan:**
1. Create Diagram: "Basic Web Application"
2. Create Node: Application Load Balancer (AWS ELB)
3. Create Cluster: "Web Tier"
4. Create Node: Web Server 1 (AWS EC2) within cluster "Web Tier"
5. Create Node: Web Server 2 (AWS EC2) within cluster "Web Tier"
6. Create Node: Database (AWS RDS)
7. Create Edge: Load Balancer → Web Servers
8. Create Edge: Web Servers → Database

**Diagram Description:**
A diagram showing an Application Load Balancer directing traffic to a cluster labeled 'Web Tier' containing two EC2 web servers. Both web servers connect to an RDS database for storage. The layout visually groups the web servers and clearly shows the flow from load balancer to web tier to database.

**Generated Python Code:**
```python
from diagrams import Diagram, Cluster
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB

with Diagram("Basic Web Application", show=False):
    alb = ELB("Application Load Balancer")
    
    with Cluster("Web Tier"):
        web_servers = [
            EC2("Web Server 1"),
            EC2("Web Server 2")
        ]
    
    database = RDS("Database")
    
    alb >> web_servers >> database
```

### /diagram – Example 2: Microservices Architecture

Request (JSON)
```json
{
  "description": "Design a microservices architecture with three services: an authentication service, a payment service, and an order service. Include an API Gateway for routing, an SQS queue for message passing between services, and a shared RDS database. Group the services in a cluster called 'Microservices'. Add CloudWatch for monitoring."
}
```

Response – `200 OK` with `Content-Type: image/png` containing the rendered diagram.<br/>
Header `X-Diagram-Plan` includes the agent's execution plan.

**Agent Execution Plan:**
1. Create Diagram: "Microservices Architecture"
2. Create Node: API Gateway (AWS API Gateway)
3. Create Cluster: "Microservices"
4. Create Node: Auth Service (AWS EC2) within cluster "Microservices"
5. Create Node: Payment Service (AWS EC2) within cluster "Microservices"
6. Create Node: Order Service (AWS EC2) within cluster "Microservices"
7. Create Node: SQS Queue (AWS SQS)
8. Create Node: Shared Database (AWS RDS)
9. Create Node: CloudWatch Monitoring (AWS CloudWatch)
10. Create Edge: API Gateway → All Services
11. Create Edge: Services → SQS Queue
12. Create Edge: Services → Database
13. Create Edge: Monitoring → Services

**Diagram Description:**
A diagram illustrating a microservices architecture: an API Gateway routes requests to a cluster labeled 'Microservices' containing authentication, payment, and order services. These services interact with an SQS queue for messaging and a shared RDS database for storage. CloudWatch is included for monitoring. The diagram visually groups the services and shows the connections between all components.

**Generated Python Code:**
```python
from diagrams import Diagram, Cluster
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import APIGateway
from diagrams.aws.integration import SQS
from diagrams.aws.management import Cloudwatch

with Diagram("Microservices Architecture", show=False):
    api_gateway = APIGateway("API Gateway")
    
    with Cluster("Microservices"):
        auth_service = EC2("Auth Service")
        payment_service = EC2("Payment Service")
        order_service = EC2("Order Service")
        services = [auth_service, payment_service, order_service]
    
    sqs_queue = SQS("SQS Queue")
    database = RDS("Shared Database")
    monitoring = Cloudwatch("CloudWatch Monitoring")
    
    api_gateway >> services
    services >> sqs_queue
    services >> database
    monitoring >> services
```

## TOOL SPECIFICATIONS:

### 1. Create Diagram Tool
- **Purpose**: Initialize a new diagram context
- **Parameters**: 
  - `name`: Diagram title
  - `outformat`: Output format (png, svg, jpg, pdf, dot)
  - `filename`: Custom filename (optional)
  - `show`: Auto-open diagram (default: false)
  - `direction`: Layout direction (TB, BT, LR, RL)
  - `graph_attr`: Custom Graphviz attributes (optional)

### 2. Create Node Tool
- **Purpose**: Create individual nodes representing system components
- **Parameters**:
  - `provider`: Cloud provider or category (aws, gcp, azure, programming, etc.)
  - `resource_type`: Specific resource type (compute, database, network, etc.)
  - `node_class`: Exact node class name
  - `label`: Display label for the node
  - `cluster`: Parent cluster name (optional)

### 3. Create Cluster Tool
- **Purpose**: Group nodes into logical clusters
- **Parameters**:
  - `name`: Cluster label
  - `parent_cluster`: Parent cluster for nesting (optional)
  - `graph_attr`: Custom cluster attributes (optional)

### 4. Create Edge Tool
- **Purpose**: Connect nodes with directional or undirected edges
- **Parameters**:
  - `from_node`: Source node identifier
  - `to_node`: Target node identifier
  - `direction`: Connection type (>>, <<, -)
  - `label`: Edge label (optional)
  - `color`: Edge color (optional)
  - `style`: Edge style (solid, dashed, dotted, bold)

### 5. Search Node Tool
- **Purpose**: Find available node types across providers
- **Parameters**:
  - `query`: Search term (e.g., "database", "load balancer", "kubernetes")
  - `provider`: Filter by provider (aws, gcp, azure, etc.)
  - `category`: Filter by category (compute, database, network, etc.)
- **Returns**: List of available node types with their import paths

## DOCUMENTATION:

- FastAPI – https://fastapi.tiangolo.com/
- OpenAI Python SDK – https://platform.openai.com/docs/api-reference
- python-dotenv – https://pypi.org/project/python-dotenv/
- Diagrams – https://diagrams.mingrammer.com/docs
- Diagrams Nodes Reference – https://diagrams.mingrammer.com/docs/nodes/aws (AWS), https://diagrams.mingrammer.com/docs/nodes/gcp (GCP), https://diagrams.mingrammer.com/docs/nodes/azure (Azure)

## OTHER CONSIDERATIONS:

- Provide `.env.example` with placeholders for `OPENAI_API_KEY`, `MODEL_NAME`, `GRAPH_RENDERER`, etc.
- README must include:
  - Local run instructions (`uv run app`)
  - Docker build & run instructions
  - Example requests and outputs
- Use `UV` CLI for environment and dependency management.
- No relative imports; set `PYTHONPATH` to project root for absolute imports.
- Mock LLM calls in unit tests to allow offline CI runs.
- Clean up temporary files created during rendering to avoid disk bloat.
- Include structured logging (Loguru) and graceful error handling with meaningful HTTP status codes.
- Aim for high test coverage with pytest and hypothesis where appropriate.
- The service must remain stateless—no user sessions and no database persistence.
- Containerize the application with both a **Dockerfile** and a **docker-compose.yml** for seamless local orchestration.
- Build agent(s) that wrap the `diagrams` package as an opaque set of tools for the LLM; do **not** instruct the LLM to reference the library directly.
  - Agents must support at least three node types across **AWS**, **Azure**, and **GCP** to provide multi-cloud diagram capabilities.
  - The service should use the agent's tools to create diagrams based on natural language descriptions, then execute the tool sequence to produce PNG/SVG images.
- Integrate with an external LLM API (e.g., **OpenAI**). All prompt logic must be explicit and visible in code—no hidden framework calls.
- (Bonus) The `/assistant` endpoint may maintain helpful context and ephemeral memory to improve multi-turn interactions.
- Provide comprehensive unit tests, ensuring LLM invocations are mocked to allow offline CI runs.
- Maintain clear project structure, modularity, and clean, well-documented code throughout the service.
- The agent must plan before executing