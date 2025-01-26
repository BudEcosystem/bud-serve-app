import os
import pytest
import warnings
from uuid import UUID
from unittest.mock import AsyncMock, patch
from fastapi import status
from fastapi.testclient import TestClient
from jose import jwt
from datetime import datetime, timedelta, UTC  # Use UTC instead of utcnow

# Suppress the crypt deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="passlib.utils")

# Set up all required environment variables before importing any application code
os.environ.update({
    # AppConfig variables
    "POSTGRES_USER": "test_user",
    "POSTGRES_PASSWORD": "test_password",
    "POSTGRES_DB": "test_db",
    "SUPER_USER_EMAIL": "test@example.com",
    "SUPER_USER_PASSWORD": "test_password",
    "DAPR_BASE_URL": "http://localhost:3500",
    "BUD_CLUSTER_APP_ID": "test_cluster_app",
    "BUD_MODEL_APP_ID": "test_model_app",
    "BUD_SIMULATOR_APP_ID": "test_simulator_app",
    "BUD_METRICS_APP_ID": "test_metrics_app",
    "BUD_NOTIFY_APP_ID": "test_notify_app",
    "APP_PORT": "8000",

    # SecretsConfig variables
    "JWT_SECRET_KEY": "test_jwt_secret_key",
    "REDIS_PASSWORD": "test_redis_password",
    "REDIS_URI": "redis://localhost:6379"
})

# Import application code after environment setup
from budapp.main import app  # Import your FastAPI app
from budapp.cluster_ops.schemas import ClusterMetricsResponse
from budapp.commons.schemas import ErrorResponse
from budapp.user_ops.schemas import User
from budapp.commons.db_utils import Session
from budapp.commons.dependencies import get_current_active_user
from budapp.cluster_ops.services import ClusterService
from budapp.cluster_ops.models import  Cluster as ClusterModel  # Import the ClusterModel
from budapp.cluster_ops.crud import ClusterDataManager  # Import the ClusterDataManager


# Helper function to generate a valid JWT token for testing
def create_test_token():
    payload = {
        "sub": "test_user",
        "email": "test@example.com",
        "exp": datetime.now(UTC) + timedelta(minutes=30)  # Use datetime.now(UTC)
    }
    secret_key = os.getenv("JWT_SECRET_KEY")
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    return token


@pytest.mark.asyncio
async def test_get_cluster_metrics():
    # Create mock objects
    mock_user = AsyncMock(spec=User)
    mock_user.is_active = True  # Ensure the user is active
    test_cluster_id = UUID("12345678-1234-5678-1234-567812345678")
    test_token = create_test_token()  # Use a valid token

    # Debug: Print the token and user
    print(f"Test Token: {test_token}")
    print(f"Mock User: {mock_user}")

    # Mock the ClusterService.get_cluster_metrics method
    mock_metrics = {
        "nodes": [
            {
                "cpu_usage": 50,
                "memory_usage": 60,
                "disk_usage": 70,
                "gpu_usage": 80,
                "hpu_usage": 90,
                "network_stats": {"in": 100, "out": 200},
            }
        ],
        "cluster_summary": {
            "total_cpu": 100,
            "total_memory": 200,
            "total_disk": 300,
            "total_gpu": 400,
            "total_hpu": 500,
        },
    }

    # Mock the db_cluster object
    mock_db_cluster = AsyncMock(spec=ClusterModel)
    mock_db_cluster.id = test_cluster_id
    mock_db_cluster.name = "Test Cluster"
    mock_db_cluster.status = "ACTIVE"  # Replace with the appropriate status enum

    # Override the get_current_active_user dependency
    def override_get_current_active_user():
        print("get_current_active_user called")  # Debugging
        return mock_user

    app.dependency_overrides[get_current_active_user] = override_get_current_active_user

    # Mock the ClusterDataManager.retrieve_by_fields method
    with patch("budapp.commons.db_utils.DataManagerUtils.retrieve_by_fields", return_value=mock_db_cluster):
        # \patch("budapp.cluster_ops.services.ClusterService.get_cluster_metrics", return_value=mock_metrics)
        try:
            # Use the TestClient to call the endpoint
            client = TestClient(app)
            response = client.get(
                f"/clusters/{test_cluster_id}/metrics",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            # Print the response for debugging
            print("\n=== Response ===")
            print(f"Status Code: {response.status_code}")
            print("Headers:")
            for key, value in response.headers.items():
                print(f"  {key}: {value}")
            print("Body:")
            print(response.json())  # Print the JSON response body

            # Verify the response
            assert response.status_code in [
                status.HTTP_200_OK
                # status.HTTP_400_BAD_REQUEST,
                # status.HTTP_500_INTERNAL_SERVER_ERROR
            ]

            if response.status_code == status.HTTP_200_OK:
                response_data = response.json()
                assert "nodes" in response_data
                assert "cluster_summary" in response_data
                assert "message" in response_data
                assert response_data["message"] == "Successfully retrieved cluster metrics"
            else:
                response_data = response.json()
                assert "message" in response_data  # Check for the "message" field in ErrorResponse
        finally:
            # Clear the dependency overrides after the test
            app.dependency_overrides.clear()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_get_cluster_metrics())
