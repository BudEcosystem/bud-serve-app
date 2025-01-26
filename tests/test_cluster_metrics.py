import pytest
from httpx import AsyncClient
from uuid import uuid4
from fastapi import status
from unittest.mock import AsyncMock, patch
from budapp.commons.config import app_settings
from budapp.main import app
from budapp.user_ops.schemas import User
from budapp.commons.dependencies import get_current_active_user
from budapp.cluster_ops.services import ClusterService
from budapp.cluster_ops.models import Cluster as ClusterModel

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def mock_user():
    user = AsyncMock(spec=User)
    user.is_active = True
    return user


@pytest.fixture
def mock_cluster():
    cluster = AsyncMock(spec=ClusterModel)
    cluster.id = uuid4()
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
    }


@pytest.fixture
def override_dependencies(mock_user):
    def override_get_current_active_user():
        return mock_user

    app.dependency_overrides[get_current_active_user] = override_get_current_active_user
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_cluster_metrics(
    async_client: AsyncClient,
    mock_user,
    mock_cluster,
    mock_metrics,
    override_dependencies,
):
    """Test the GET /clusters/{cluster_id}/metrics endpoint."""
    with patch(
        "budapp.commons.db_utils.DataManagerUtils.retrieve_by_fields", return_value=mock_cluster
    ), patch.object(ClusterService, "get_cluster_metrics", return_value=mock_metrics):
        response = await async_client.get(f"/clusters/{mock_cluster.id}/metrics")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["object"] == "cluster.metrics"
        assert "nodes" in response.json()
        assert "cluster_summary" in response.json()
        assert response.json()["message"] == "Successfully retrieved cluster metrics"


@pytest.mark.asyncio
async def test_get_cluster_metrics_invalid_cluster_id(
    async_client: AsyncClient, mock_user, override_dependencies
):
    """Test the GET /clusters/{cluster_id}/metrics endpoint with an invalid cluster ID."""
    response = await async_client.get(f"/clusters/invalid-uuid/metrics")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["object"] == "error"
    assert "message" in response.json()


@pytest.mark.asyncio
async def test_get_cluster_metrics_unauthorized(async_client: AsyncClient):
    """Test the GET /clusters/{cluster_id}/metrics endpoint with an invalid token."""
    cluster_id = uuid4()
    response = await async_client.get(
        f"/clusters/{cluster_id}/metrics",
        headers={"Authorization": "Bearer invalid_token"},
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["object"] == "error"
    assert "message" in response.json()
