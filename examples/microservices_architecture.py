#!/usr/bin/env python3
"""
Example: Microservices Architecture Diagram
Demonstrates creating a diagram showing a microservices architecture with three services:
authentication, payment, and order services, with API Gateway, SQS, and shared RDS database.

Diagram Description:
A diagram illustrating a microservices architecture: an API Gateway routes requests to a cluster 
labeled 'Microservices' containing authentication, payment, and order services. These services 
interact with an SQS queue for messaging and a shared RDS database for storage. CloudWatch is 
included for monitoring. The diagram visually groups the services and shows the connections 
between all components.
"""

from diagrams import Diagram, Cluster
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import APIGateway
from diagrams.aws.integration import SQS
from diagrams.aws.management import Cloudwatch

def create_microservices_architecture():
    """Create a microservices architecture diagram."""
    with Diagram("Microservices Architecture", show=False, filename="microservices_arch"):
        # API Gateway
        api_gateway = APIGateway("API Gateway")
        
        # Microservices cluster
        with Cluster("Microservices"):
            auth_service = EC2("Auth Service")
            payment_service = EC2("Payment Service")
            order_service = EC2("Order Service")
            services = [auth_service, payment_service, order_service]
        
        # Shared infrastructure
        sqs_queue = SQS("SQS Queue")
        database = RDS("Shared RDS")
        monitoring = Cloudwatch("Monitoring")
        
        # Connect components
        api_gateway >> services
        services >> sqs_queue
        services >> database
        monitoring >> services

if __name__ == "__main__":
    create_microservices_architecture()
    print("Microservices architecture diagram created successfully!") 