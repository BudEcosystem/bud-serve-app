import os
import pytest
from uuid import UUID
from unittest.mock import AsyncMock, patch
from fastapi import status
from fastapi.testclient import TestClient

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

@pytest.mark.asyncio
async def test_get_cluster_metrics():
    # Create mock objects
    mock_user = AsyncMock(spec=User)
    mock_session = AsyncMock(spec=Session)
    test_cluster_id = UUID("12345678-1234-5678-1234-567812345678")
    test_token = "test_jwt_token"

    # Patch the dependencies with the correct path to get_current_active_user
    with patch("budapp.commons.dependencies.get_current_active_user", return_value=mock_user), \
         patch("budapp.commons.dependencies.get_session", return_value=mock_session):

        # Use the TestClient to call the endpoint
        client = TestClient(app)
        response = client.get(f"/clusters/{test_cluster_id}/metrics", headers={"Authorization": f"Bearer {test_token}"})

        # Verify the response
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]

        if response.status_code == status.HTTP_200_OK:
            response_data = response.json()
            assert "cpu_usage" in response_data
            assert "memory_usage" in response_data
            assert "disk_usage" in response_data
            assert "gpu_usage" in response_data
            assert "hpu_usage" in response_data
            assert "network_stats" in response_data
        else:
            response_data = response.json()
            assert "detail" in response_data
            assert "message" in response_data

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_get_cluster_metrics())
