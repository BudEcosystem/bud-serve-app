import os
from pathlib import Path
from dotenv import load_dotenv

# Load test environment variables before any other imports
env_path = Path(__file__).parent.parent / '.env.test'
load_dotenv(env_path, override=True)

import pytest
from fastapi.testclient import TestClient
from uuid import UUID
from fastapi import status
from unittest.mock import AsyncMock, patch
from budapp.commons.config import app_settings
from budapp.main import app
from budapp.user_ops.schemas import User
from budapp.commons.dependencies import get_current_active_user
from budapp.cluster_ops.services import ClusterService
from budapp.cluster_ops.models import Cluster as ClusterModel

pytest_plugins = ("pytest_asyncio",)

TEST_CLUSTER_ID = UUID('7b69d2f3-5524-484a-9a22-4ad3e4f67639')


@pytest.fixture
def mock_user():
    user = AsyncMock(spec=User)
    user.is_active = True
    return user

@pytest.fixture(autouse=True)
def setup_test_env():
    """Fixture to set up test environment variables."""
    # Store original env vars
    original_env = dict(os.environ)
    
    # Set up test environment
    load_dotenv(env_path, override=True)
    
    yield
    
    # Restore original env vars
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_client():
    """Fixture that creates a test client for testing."""
    return TestClient(app)


@pytest.fixture
def mock_cluster():
    cluster = AsyncMock(spec=ClusterModel)
    cluster.id = TEST_CLUSTER_ID
    cluster.name = "Test Cluster"
    cluster.status = "ACTIVE"
    return cluster


@pytest.fixture
def mock_metrics():
    return {
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
        "historical_data": {
            "cpu_usage": [
                {"timestamp": 1234567890, "value": 45.5},
                {"timestamp": 1234567900, "value": 50.2}
            ]
        },
        "time_range": "today"
    }


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    return AsyncMock()


@pytest.fixture
def override_dependencies(mock_user, mock_db_session):
    """Override FastAPI dependencies."""
    def override_get_current_active_user():
        return mock_user

    def override_get_session():
        return mock_db_session

    app.dependency_overrides[get_current_active_user] = override_get_current_active_user
    app.dependency_overrides["get_session"] = override_get_session
    yield
    app.dependency_overrides.clear()


@pytest.mark.parametrize("time_range", ["today","7days","month"])
def test_get_cluster_metrics(
    test_client: TestClient,
    mock_user,
    mock_cluster,
    mock_metrics,
    override_dependencies,
    time_range: str,
):
    """Test the GET /clusters/{cluster_id}/metrics endpoint with different time ranges."""
    with patch(
        "budapp.commons.db_utils.DataManagerUtils.retrieve_by_fields", return_value=mock_cluster
    ):
        response = test_client.get(
            f"/clusters/{TEST_CLUSTER_ID}/metrics",
            params={"time_range": time_range}
        )

        with open(f"cluster_metrics_response_{time_range}.json", "w") as file:
            file.write(response.text)


# def test_get_cluster_metrics_invalid_time_range(
#     test_client: TestClient,
#     mock_user,
#     mock_cluster,
#     override_dependencies,
# ):
#     """Test the GET /clusters/{cluster_id}/metrics endpoint with an invalid time range."""
#     response = test_client.get(
#         f"/clusters/{TEST_CLUSTER_ID}/metrics",
#         params={"time_range": "invalid_range"}
#     )

#     assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
#     assert response.json()["object"] == "error"
#     assert "time_range" in response.json()["detail"][0]["loc"]


# def test_get_cluster_metrics_invalid_cluster_id(
#     test_client: TestClient,
#     mock_user,
#     override_dependencies
# ):
#     """Test the GET /clusters/{cluster_id}/metrics endpoint with an invalid cluster ID."""
#     response = test_client.get(f"/clusters/invalid-uuid/metrics")

#     assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
#     assert response.json()["object"] == "error"
#     assert "message" in response.json()


# def test_get_cluster_metrics_unauthorized(test_client: TestClient):
#     """Test the GET /clusters/{cluster_id}/metrics endpoint with an invalid token."""
#     response = test_client.get(
#         f"/clusters/{TEST_CLUSTER_ID}/metrics",
#         headers={"Authorization": "Bearer invalid_token"},
#     )

#     assert response.status_code == status.HTTP_401_UNAUTHORIZED
#     assert response.json()["object"] == "error"
#     assert "message" in response.json()


# def test_get_cluster_metrics_not_found(
#     test_client: TestClient,
#     mock_user,
#     override_dependencies,
# ):
#     """Test the GET /clusters/{cluster_id}/metrics endpoint with a non-existent cluster."""
#     with patch(
#         "budapp.commons.db_utils.DataManagerUtils.retrieve_by_fields",
#         return_value=None
#     ):
#         response = test_client.get(f"/clusters/{TEST_CLUSTER_ID}/metrics")

#         assert response.status_code == status.HTTP_404_NOT_FOUND
#         assert response.json()["object"] == "error"
#         assert response.json()["message"] == "Cluster not found"


# def test_get_cluster_metrics_prometheus_error(
#     test_client: TestClient,
#     mock_user,
#     mock_cluster,
#     override_dependencies,
# ):
#     """Test handling of Prometheus service unavailability."""
#     with patch(
#         "budapp.commons.db_utils.DataManagerUtils.retrieve_by_fields",
#         return_value=mock_cluster
#     ), patch.object(
#         ClusterService,
#         "get_cluster_metrics",
#         return_value=None
#     ):
#         response = test_client.get(f"/clusters/{TEST_CLUSTER_ID}/metrics")

#         assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
#         assert response.json()["object"] == "error"
#         assert "Failed to fetch metrics from Prometheus" in response.json()["message"]
