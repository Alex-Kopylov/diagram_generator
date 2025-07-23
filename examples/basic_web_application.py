#!/usr/bin/env python3
"""
Example: Basic Web Application Diagram
Demonstrates creating a diagram showing a basic web application with an Application Load Balancer,
two EC2 instances for the web servers, and an RDS database for storage.

Diagram Description:
A diagram showing an Application Load Balancer directing traffic to a cluster labeled 'Web Tier' 
containing two EC2 web servers. Both web servers connect to an RDS database for storage. 
The layout visually groups the web servers and clearly shows the flow from load balancer to web tier to database.
"""

from diagrams import Diagram
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB
from diagrams import Cluster

def create_basic_web_application():
    """Create a basic web application architecture diagram."""
    with Diagram("Basic Web Application", show=False, filename="basic_web_app"):
        # Application Load Balancer
        alb = ELB("Application Load Balancer")
        
        # Web Tier cluster with EC2 instances
        with Cluster("Web Tier"):
            web_servers = [
                EC2("Web Server 1"),
                EC2("Web Server 2")
            ]
        
        # RDS Database
        database = RDS("Database")
        
        # Connect components
        alb >> web_servers[0]
        alb >> web_servers[1]
        web_servers[0] >> database
        web_servers[1] >> database

if __name__ == "__main__":
    create_basic_web_application()
    print("Basic web application diagram created successfully!") 