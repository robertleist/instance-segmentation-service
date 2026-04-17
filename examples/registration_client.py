#!/usr/bin/env python
"""
Example client for registering a service with the Coral template microservice.

This demonstrates how the main backend would register the service and configure
Celery, MLFlow, and authentication.
"""

import requests
import json
import os
from typing import Optional

class ServiceRegistrationClient:
    def __init__(self, service_url: str):
        """Initialize the client."""
        self.service_url = service_url.rstrip("/")

    def register_service(
        self,
        registration_token: str,
        celery_broker_url: str,
        mlflow_tracking_uri: str,
        backend_url: str = "http://localhost:8000",
        service_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """
        Register the service with the main backend configuration.
        
        Args:
            registration_token: Token to authenticate registration (should match SERVICE_REGISTRATION_TOKEN env var)
            celery_broker_url: URL for the Celery broker (e.g., redis://localhost:6379/0)
            mlflow_tracking_uri: URL for the MLFlow tracking server (e.g., http://localhost:5000)
            backend_url: URL of the main backend
            service_name: Optional name for this service instance
            api_key: Optional API key to set for service authentication
            
        Returns:
            Registration response dict
        """
        payload = {
            "backend_url": backend_url,
            "celery_broker_url": celery_broker_url,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "registration_token": registration_token,
            "service_name": service_name,
            "api_key": api_key,
        }
        
        response = requests.post(
            f"{self.service_url}/register",
            json=payload,
        )
        
        if response.status_code != 200:
            print(f"Registration failed: {response.status_code}")
            print(response.text)
            return {"success": False, "error": response.text}
        
        return response.json()

    def call_with_api_key(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[dict] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Make an authenticated request to the service."""
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        
        url = f"{self.service_url}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return response.json()

if __name__ == "__main__":
    # Example usage
    client = ServiceRegistrationClient("http://localhost:8000")
    
    # Register the service
    result = client.register_service(
        registration_token="default-secret",  # Should match SERVICE_REGISTRATION_TOKEN
        celery_broker_url="redis://localhost:6379/0",
        mlflow_tracking_uri="http://localhost:5000",
        api_key="my-service-key-12345",
    )
    
    print("Registration result:")
    print(json.dumps(result, indent=2))
    
    if result.get("success"):
        # Now you can call protected endpoints with the API key
        models = client.call_with_api_key(
            "/get_available_models",
            api_key="my-service-key-12345"
        )
        print("\nAvailable models:")
        print(json.dumps(models, indent=2))