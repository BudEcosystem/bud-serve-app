# tests/test_cluster_metrics.py

import pytest
from unittest.mock import Mock, patch
from uuid import UUID
from datetime import datetime

from fastapi import status
from sqlalchemy.orm import Session

from budapp.cluster_ops.models import Cluster as ClusterModel
from budapp.cluster_ops.services import ClusterService
from budapp.cluster_ops.schemas import ClusterMetricsResponse
from budapp.commons.exceptions import ClientException
from budapp.commons.constants import ClusterStatusEnum

# Mock data
MOCK_CLUSTER_ID = UUID("123e4567-e89b-12d3-a456-426614174000")

MOCK_DB_CLUSTER = ClusterModel(
    id=MOCK_CLUSTER_ID,
    name="test-cluster",
    ingress_url="http://test-cluster.com",
    status=ClusterStatusEnum.AVAILABLE,
    icon="test-icon",
    cpu_count=4,
    gpu_count=2,
    hpu_count=1,
    cpu_total_workers=8,
    cpu_available_workers=4,
    gpu_total_workers=4,
    gpu_available_workers=2,
    hpu_total_workers=2,
    hpu_available_workers=1,
    created_by=UUID("123e4567-e89b-12d3-a456-426614174001"),
    cluster_id=UUID("123e4567-e89b-12d3-a456-426614174002"),
    created_at=datetime.utcnow(),
    modified_at=datetime.utcnow(),
    status_sync_at=datetime.utcnow()
)

MOCK_PROMETHEUS_RESPONSE = {
    "test-cluster": {
        "nodes": {
            "node1": {
                "memory": {
                    "memory_total": 128.0,
                    "memory_used": 64.0,
                    "memory_available": 64.0,
                    "memory_usage_percent": 50.0
                },
                "cpu": {
                    "cpu_cores": 32,
                    "cpu_usage_percent": 45.5
                },
                "disk": {
                    "paths": {
                        "/": {
                            "device": "/dev/sda1",
                            "fstype": "ext4",
                            "total_gib": 500.0,
                            "available_gib": 250.0,
                            "used_gib": 250.0,
                            "usage_percent": 50.0
                        }
                    }
                },
                "gpu": {
                    "gpu_count": 4,
                    "gpu_memory_total": 32.0,
                    "gpu_memory_used": 16.0,
                    "gpu_memory_available": 16.0,
                    "gpu_memory_usage_percent": 50.0,
                    "gpu_utilization": 75.0
                },
                "hpu": {
                    "hpu_count": 2,
                    "hpu_memory_total": 16.0,
                    "hpu_memory_used": 8.0,
                    "hpu_memory_available": 8.0,
                    "hpu_memory_usage_percent": 50.0,
                    "hpu_utilization": 60.0
                },
                "network": {
                    "interfaces": {
                        "eth0": {
                            "network_receive_bytes_total": 100.0,
                            "network_transmit_bytes_total": 100.0,
                            "network_receive_errors": 0,
                            "network_transmit_errors": 0
                        }
                    },
                    "summary": {
                        "total_receive_mbps": 100.0,
                        "total_transmit_mbps": 100.0,
                        "total_bandwidth_mbps": 200.0,
                        "total_errors": 0
                    }
                }
            }
        },
        "cluster_summary": {
            "total_nodes": 1,
            "memory": {
                "total_gib": 128.0,
                "used_gib": 64.0,
                "available_gib": 64.0,
                "usage_percent": 50.0
            },
            "disk": {
                "total_gib": 500.0,
                "used_gib": 250.0,
                "available_gib": 250.0,
                "usage_percent": 50.0
            },
            "gpu": {
                "total_memory_gib": 32.0,
                "used_memory_gib": 16.0,
                "available_memory_gib": 16.0,
                "memory_usage_percent": 50.0,
                "average_utilization_percent": 75.0
            },
            "hpu": {
                "total_memory_gib": 16.0,
                "used_memory_gib": 8.0,
                "available_memory_gib": 8.0,
                "memory_usage_percent": 50.0,
                "average_utilization_percent": 60.0
            },
            "cpu": {
                "average_usage_percent": 45.5
            },
            "network": {
                "total_receive_mbps": 100.0,
                "total_transmit_mbps": 100.0,
                "total_errors": 0
            }
        }
    }
}

@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return Mock(spec=Session)

@pytest.fixture
def cluster_service(mock_session):
    """Create a ClusterService instance with a mock session."""
    return ClusterService(mock_session)

@pytest.mark.asyncio
async def test_get_cluster_metrics_success(cluster_service):
    """Test successful retrieval of cluster metrics."""
    # Mock the database query
    cluster_service.session.query().filter().first.return_value = MOCK_DB_CLUSTER

    # Mock the ClusterDataManager's retrieve_by_fields method
    with patch('budapp.cluster_ops.crud.ClusterDataManager.retrieve_by_fields') as mock_retrieve:
        mock_retrieve.return_value = MOCK_DB_CLUSTER

        # Mock the ClusterMetricsFetcher
        with patch('budapp.cluster_ops.utils.ClusterMetricsFetcher') as MockFetcher:
            mock_fetcher_instance = MockFetcher.return_value
            mock_fetcher_instance.get_cluster_metrics.return_value = MOCK_PROMETHEUS_RESPONSE

            # Call the service method
            metrics = await cluster_service.get_cluster_metrics(MOCK_CLUSTER_ID)

            # Validate the response
            assert metrics is not None
            assert "nodes" in metrics
            assert "cluster_summary" in metrics

            # Validate specific metrics
            summary = metrics["cluster_summary"]
            assert summary["total_nodes"] == 1
            assert summary["cpu"]["average_usage_percent"] == 45.5
            assert summary["memory"]["usage_percent"] == 50.0

@pytest.mark.asyncio
async def test_get_cluster_metrics_not_found(cluster_service):
    """Test cluster metrics retrieval when cluster is not found."""
    # Mock the ClusterDataManager to return None
    with patch('budapp.cluster_ops.crud.ClusterDataManager.retrieve_by_fields') as mock_retrieve:
        mock_retrieve.side_effect = ClientException("Cluster not found")

        with pytest.raises(ClientException) as exc_info:
            await cluster_service.get_cluster_metrics(MOCK_CLUSTER_ID)

        assert "Cluster not found" in str(exc_info.value)

@pytest.mark.asyncio
async def test_get_cluster_metrics_prometheus_error(cluster_service):
    """Test handling of Prometheus errors."""
    # Mock the database query to return a valid cluster
    with patch('budapp.cluster_ops.crud.ClusterDataManager.retrieve_by_fields') as mock_retrieve:
        mock_retrieve.return_value = MOCK_DB_CLUSTER

        # Mock the ClusterMetricsFetcher to raise an exception
        with patch('budapp.cluster_ops.utils.ClusterMetricsFetcher') as MockFetcher:
            mock_fetcher_instance = MockFetcher.return_value
            mock_fetcher_instance.get_cluster_metrics.return_value = None

            with pytest.raises(ClientException) as exc_info:
                await cluster_service.get_cluster_metrics(MOCK_CLUSTER_ID)

            assert "Failed to fetch metrics from Prometheus" in str(exc_info.value)

# Test the API endpoint
from fastapi.testclient import TestClient
from budapp.main import app

client = TestClient(app)

def test_get_cluster_metrics_endpoint(monkeypatch):
    """Test the GET /clusters/{cluster_id}/metrics endpoint."""
    # Mock the authentication dependency
    async def mock_get_current_active_user():
        return {"id": "user123", "email": "test@example.com"}

    app.dependency_overrides[get_current_active_user] = mock_get_current_active_user

    # Mock the ClusterService
    async def mock_get_cluster_metrics(*args, **kwargs):
        return MOCK_PROMETHEUS_RESPONSE["test-cluster"]

    monkeypatch.setattr(
        "budapp.cluster_ops.services.ClusterService.get_cluster_metrics",
        mock_get_cluster_metrics
    )

    response = client.get(f"/clusters/{MOCK_CLUSTER_ID}/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "cluster.metrics"
    assert "nodes" in data
    assert "cluster_summary" in data

def test_get_cluster_metrics_endpoint_error(monkeypatch):
    """Test error handling in the metrics endpoint."""
    # Mock the authentication dependency
    async def mock_get_current_active_user():
        return {"id": "user123", "email": "test@example.com"}

    app.dependency_overrides[get_current_active_user] = mock_get_current_active_user

    # Mock the ClusterService to raise an error
    async def mock_get_cluster_metrics(*args, **kwargs):
        raise ClientException("Test error")

    monkeypatch.setattr(
        "budapp.cluster_ops.services.ClusterService.get_cluster_metrics",
        mock_get_cluster_metrics
    )

    response = client.get(f"/clusters/{MOCK_CLUSTER_ID}/metrics")

    assert response.status_code == 400
    data = response.json()
    assert data["message"] == "Test error"
