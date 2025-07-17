#  -----------------------------------------------------------------------------
#  Copyright (c) 2024 Bud Ecosystem Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  -----------------------------------------------------------------------------

"""Tests for model deployment functionality."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from budapp.commons.constants import (
    EndpointStatusEnum,
    ModelProviderTypeEnum,
    ModelStatusEnum,
)
from budapp.endpoint_ops.models import Endpoint as EndpointModel
from budapp.model_ops.models import CloudModel, Model
from budapp.model_ops.schemas import DeploymentTemplateCreate
from budapp.model_ops.services import ModelService


@pytest.mark.asyncio
async def test_create_endpoint_directly_for_cloud_model():
    """Test direct endpoint creation for cloud models."""
    # Setup
    model_id = uuid4()
    project_id = uuid4()
    workflow_id = uuid4()
    user_id = uuid4()
    credential_id = uuid4()
    cloud_model_id = uuid4()

    # Mock model data
    mock_model = MagicMock(spec=Model)
    mock_model.id = model_id
    mock_model.provider_type = ModelProviderTypeEnum.CLOUD_MODEL
    mock_model.status = ModelStatusEnum.ACTIVE
    mock_model.source = "openai"  # For proxy cache and cloud model lookup
    mock_model.uri = "gpt-4"
    mock_model.provider_id = uuid4()

    # Mock cloud model data
    mock_cloud_model = MagicMock(spec=CloudModel)
    mock_cloud_model.id = cloud_model_id
    mock_cloud_model.supported_endpoints = ["chat", "completion"]

    # Mock deployment config
    deploy_config = DeploymentTemplateCreate(
        concurrent_requests=10, avg_sequence_length=100, avg_context_length=1000, replicas=1
    )

    # Mock endpoint
    mock_endpoint = MagicMock(spec=EndpointModel)
    mock_endpoint.id = uuid4()
    mock_endpoint.url = "https://example.com/model-namespace"
    mock_endpoint.status = EndpointStatusEnum.RUNNING

    # Create service with mocked session
    mock_session = AsyncMock()
    service = ModelService(session=mock_session)

    # Mock database operations
    with patch("budapp.model_ops.crud.ModelDataManager") as mock_model_manager, patch(
        "budapp.model_ops.crud.CloudModelDataManager"
    ) as mock_cloud_model_manager, patch(
        "budapp.endpoint_ops.crud.EndpointDataManager"
    ) as mock_endpoint_manager, patch("budapp.commons.config.app_settings") as mock_settings, patch(
        "budapp.model_ops.services.EndpointService"
    ) as mock_endpoint_service, patch("budapp.credential_ops.services.CredentialService") as mock_credential_service:
        # Configure mocks
        mock_settings.base_deployment_url = "https://example.com"

        mock_model_manager.return_value.retrieve_by_fields = AsyncMock(return_value=mock_model)
        mock_cloud_model_manager.return_value.retrieve_by_fields = AsyncMock(return_value=mock_cloud_model)
        mock_endpoint_manager.return_value.insert_one = AsyncMock(return_value=mock_endpoint)

        # Configure endpoint service and credential service mocks
        mock_endpoint_service_instance = AsyncMock()
        mock_endpoint_service.return_value = mock_endpoint_service_instance
        mock_credential_service_instance = AsyncMock()
        mock_credential_service.return_value = mock_credential_service_instance

        # Execute
        result = await service._create_endpoint_directly(
            model_id=model_id,
            project_id=project_id,
            cluster_id=None,  # Test with no cluster for cloud models
            endpoint_name="test-endpoint",
            deploy_config=deploy_config,
            workflow_id=workflow_id,
            current_user_id=user_id,
            credential_id=credential_id,
        )

        # Verify
        assert result.id == mock_endpoint.id
        assert result.status == EndpointStatusEnum.RUNNING

        # Verify model was retrieved
        mock_model_manager.return_value.retrieve_by_fields.assert_called_once()

        # Verify cloud model was retrieved
        mock_cloud_model_manager.return_value.retrieve_by_fields.assert_called_once()

        # Verify endpoint was created
        mock_endpoint_manager.return_value.insert_one.assert_called_once()
        endpoint_model = mock_endpoint_manager.return_value.insert_one.call_args[0][0]

        # Verify endpoint model is correct type and has expected attributes
        assert hasattr(endpoint_model, "project_id")
        assert hasattr(endpoint_model, "model_id")
        assert hasattr(endpoint_model, "cluster_id")
        assert hasattr(endpoint_model, "name")
        assert hasattr(endpoint_model, "status")
        assert hasattr(endpoint_model, "credential_id")
        assert hasattr(endpoint_model, "supported_endpoints")
        assert hasattr(endpoint_model, "active_replicas")
        assert hasattr(endpoint_model, "total_replicas")
        assert hasattr(endpoint_model, "number_of_nodes")
        assert hasattr(endpoint_model, "namespace")
        assert hasattr(endpoint_model, "url")

        # Verify proxy cache was updated
        mock_endpoint_service_instance.add_model_to_proxy_cache.assert_called_once()
        mock_credential_service_instance.update_proxy_cache.assert_called_once_with(project_id)


@pytest.mark.asyncio
async def test_create_endpoint_directly_raises_for_non_cloud_model():
    """Test that direct endpoint creation raises error for non-cloud models."""
    # Setup
    model_id = uuid4()
    mock_model = MagicMock(spec=Model)
    mock_model.provider_type = ModelProviderTypeEnum.HUGGING_FACE  # Not a cloud model

    # Create service
    mock_session = AsyncMock()
    service = ModelService(session=mock_session)

    with patch("budapp.model_ops.crud.ModelDataManager") as mock_model_manager:
        mock_model_manager.return_value.retrieve_by_fields = AsyncMock(return_value=mock_model)

        # Execute and verify exception
        with pytest.raises(Exception) as exc_info:
            await service._create_endpoint_directly(
                model_id=model_id,
                project_id=uuid4(),
                cluster_id=uuid4(),
                endpoint_name="test",
                deploy_config=DeploymentTemplateCreate(
                    concurrent_requests=10, avg_sequence_length=100, avg_context_length=1000
                ),
                workflow_id=uuid4(),
                current_user_id=uuid4(),
            )

        assert "Direct endpoint creation is only supported for cloud models" in str(exc_info.value)


@pytest.mark.asyncio
async def test_deploy_model_by_step_uses_direct_creation_for_cloud_models():
    """Test that deploy_model_by_step uses direct creation for cloud models."""
    # This test would be more complex and require mocking the full workflow
    # For now, we'll create a basic structure
    pass  # TODO: Implement full integration test
