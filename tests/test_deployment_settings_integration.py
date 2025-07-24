#  -----------------------------------------------------------------------------
#  Copyright (c) 2024 Bud Ecosystem Inc.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  -----------------------------------------------------------------------------

"""Integration tests for deployment settings functionality."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from budapp.commons.constants import PermissionEnum
from budapp.endpoint_ops.schemas import (
    DeploymentSettingsConfig,
    FallbackConfig,
    RateLimitConfig,
    RetryConfig,
)


@pytest.mark.asyncio
async def test_deployment_settings_api_flow():
    """Test complete API flow for deployment settings."""
    # Mock user and endpoint data
    user_id = uuid4()
    endpoint_id = uuid4()
    project_id = uuid4()

    # Mock user with proper permissions
    mock_user = MagicMock()
    mock_user.id = user_id
    mock_user.is_active = True

    # Mock endpoint
    mock_endpoint = MagicMock()
    mock_endpoint.id = endpoint_id
    mock_endpoint.project_id = project_id
    mock_endpoint.name = "test-endpoint"
    mock_endpoint.deployment_config = {}

    with patch("budapp.commons.dependencies.get_current_active_user") as mock_get_user, \
         patch("budapp.commons.dependencies.get_session") as mock_get_session, \
         patch("budapp.commons.permission_handler.get_user_permissions", return_value=[PermissionEnum.ENDPOINT_VIEW, PermissionEnum.ENDPOINT_MANAGE]), \
         patch("budapp.endpoint_ops.services.EndpointDataManager") as mock_endpoint_manager_class, \
         patch("budapp.endpoint_ops.services.ModelDataManager") as mock_model_manager_class, \
         patch("budapp.endpoint_ops.services.RedisService") as mock_redis_class, \
         patch("budapp.endpoint_ops.services.BudNotifyService") as mock_notify_class:

        # Set up mocks
        mock_get_user.return_value = mock_user
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_endpoint_manager = MagicMock()
        mock_endpoint_manager.retrieve_by_field = AsyncMock(return_value=mock_endpoint)
        mock_endpoint_manager.update_by_fields = AsyncMock()
        mock_endpoint_manager_class.return_value = mock_endpoint_manager

        mock_model_manager = MagicMock()
        mock_model_manager.retrieve_many_by_fields = AsyncMock(return_value=[])
        mock_model_manager.retrieve_by_field = AsyncMock(return_value=None)
        mock_model_manager_class.return_value = mock_model_manager

        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()
        mock_redis_class.return_value = mock_redis

        mock_notify_class.send_notification = AsyncMock()

        # Import app after mocks are set up
        from budapp.main import app

        client = TestClient(app)

        # Test 1: Get deployment settings (should return defaults)
        response = client.get(
            f"/endpoints/{endpoint_id}/deployment-settings",
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["result"]["endpoint_id"] == str(endpoint_id)
        assert data["result"]["deployment_settings"]["rate_limits"] is None
        assert data["result"]["deployment_settings"]["retry_config"] is None
        assert data["result"]["deployment_settings"]["fallback_config"] is None

        # Test 2: Update deployment settings
        update_payload = {
            "rate_limits": {
                "algorithm": "token_bucket",
                "requests_per_second": 20,
                "burst_size": 30,
                "enabled": True,
            },
            "retry_config": {
                "num_retries": 3,
                "max_delay_s": 5.0,
            },
        }

        # Update endpoint config for next retrieval
        mock_endpoint.deployment_config = {
            "deployment_settings": update_payload
        }

        response = client.put(
            f"/endpoints/{endpoint_id}/deployment-settings",
            json=update_payload,
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["result"]["deployment_settings"]["rate_limits"]["requests_per_second"] == 20
        assert data["result"]["deployment_settings"]["retry_config"]["num_retries"] == 3

        # Verify cache was updated
        mock_redis.set.assert_called()


@pytest.mark.asyncio
async def test_deployment_settings_validation_errors():
    """Test validation errors for deployment settings."""
    from budapp.main import app

    client = TestClient(app)
    endpoint_id = uuid4()

    with patch("budapp.commons.dependencies.get_current_active_user") as mock_get_user, \
         patch("budapp.commons.dependencies.get_session"), \
         patch("budapp.commons.permission_handler.get_user_permissions", return_value=[PermissionEnum.ENDPOINT_MANAGE]):

        mock_user = MagicMock()
        mock_user.is_active = True
        mock_get_user.return_value = mock_user

        # Test invalid rate limit algorithm
        invalid_payload = {
            "rate_limits": {
                "algorithm": "invalid_algorithm",
            }
        }

        response = client.put(
            f"/endpoints/{endpoint_id}/deployment-settings",
            json=invalid_payload,
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test negative retry count
        invalid_payload = {
            "retry_config": {
                "num_retries": -1,
            }
        }

        response = client.put(
            f"/endpoints/{endpoint_id}/deployment-settings",
            json=invalid_payload,
            headers={"Authorization": "Bearer test-token"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY