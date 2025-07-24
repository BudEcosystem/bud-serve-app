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

"""Tests for cloud model delete functionality."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from budapp.commons.constants import (
    EndpointStatusEnum,
    ModelProviderTypeEnum,
    WorkflowStatusEnum,
)
from budapp.endpoint_ops.services import EndpointService


class MockProject:
    """Mock project object with required attributes."""
    def __init__(self, id, name):
        self.id = id
        self.name = name


class MockModel:
    """Mock model object with required attributes."""
    def __init__(self, id, provider_type, icon, provider_id=None):
        self.id = id
        self.provider_type = provider_type
        self.icon = icon
        self.provider_id = provider_id
        self.source = "openai"
        self.uri = "gpt-4"


class MockCluster:
    """Mock cluster object with required attributes."""
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id


class MockEndpoint:
    """Mock endpoint object with required attributes."""
    def __init__(self, id, project_id, model_id, model, project, name, status, cluster=None, namespace=None):
        self.id = id
        self.project_id = project_id
        self.model_id = model_id
        self.model = model
        self.project = project
        self.name = name
        self.status = status
        self.cluster = cluster
        self.namespace = namespace


@pytest.mark.asyncio
async def test_delete_cloud_model_endpoint_immediate_deletion():
    """Test that cloud model endpoints without clusters are deleted immediately."""
    # Setup
    endpoint_id = uuid4()
    user_id = uuid4()
    project_id = uuid4()
    model_id = uuid4()
    workflow_id = uuid4()

    # Create mock objects with actual attributes
    mock_project = MockProject(id=project_id, name="test-project")
    mock_model = MockModel(
        id=model_id,
        provider_type=ModelProviderTypeEnum.CLOUD_MODEL,
        icon="test-icon",
        provider_id=None
    )
    mock_endpoint = MockEndpoint(
        id=endpoint_id,
        project_id=project_id,
        model_id=model_id,
        model=mock_model,
        project=mock_project,
        name="test-endpoint",
        status=EndpointStatusEnum.RUNNING,
        cluster=None
    )

    # Mock workflow data
    mock_workflow = MagicMock()
    mock_workflow.id = workflow_id
    mock_workflow.status = WorkflowStatusEnum.IN_PROGRESS
    mock_workflow.created_by = user_id

    # Create service with mocked session
    mock_session = MagicMock()
    service = EndpointService(session=mock_session)

    # Mock database operations and services
    with patch("budapp.endpoint_ops.crud.EndpointDataManager") as mock_endpoint_manager, \
         patch("budapp.workflow_ops.crud.WorkflowDataManager") as mock_workflow_manager, \
         patch("budapp.shared.notification_service.BudNotifyService") as mock_notify_service, \
         patch("budapp.credential_ops.services.CredentialService") as mock_credential_service, \
         patch("budapp.model_ops.crud.ProviderDataManager") as mock_provider_manager:
        
        # Configure endpoint manager
        mock_endpoint_manager_instance = AsyncMock()
        mock_endpoint_manager.return_value = mock_endpoint_manager_instance
        mock_endpoint_manager_instance.retrieve_by_fields.return_value = mock_endpoint
        mock_endpoint_manager_instance.update_by_fields.return_value = mock_endpoint

        # Configure workflow manager
        mock_workflow_manager_instance = AsyncMock()
        mock_workflow_manager.return_value = mock_workflow_manager_instance
        # Update the workflow status to COMPLETED when it's updated
        async def mock_update_workflow(workflow, updates):
            if "status" in updates:
                workflow.status = updates["status"]
            return workflow
        mock_workflow_manager_instance.update_by_fields.side_effect = mock_update_workflow

        # Mock notification service
        mock_notify_service_instance = AsyncMock()
        mock_notify_service.return_value = mock_notify_service_instance
        mock_notify_service_instance.send_notification.return_value = None

        # Mock credential service
        mock_credential_service_instance = AsyncMock()
        mock_credential_service.return_value = mock_credential_service_instance
        mock_credential_service_instance.update_proxy_cache.return_value = None

        # Mock Redis operations
        with patch.object(service, 'delete_model_from_proxy_cache') as mock_delete_proxy:
            mock_delete_proxy.return_value = None

            # Mock workflow creation
            with patch("budapp.workflow_ops.services.WorkflowService") as mock_workflow_service:
                mock_workflow_service_instance = AsyncMock()
                mock_workflow_service.return_value = mock_workflow_service_instance
                mock_workflow_service_instance.retrieve_or_create_workflow.return_value = mock_workflow

                # Execute
                result = await service.delete_endpoint(endpoint_id, user_id)

                # Verify
                assert result.id == workflow_id
                assert result.status == WorkflowStatusEnum.COMPLETED

                # Verify endpoint was retrieved
                mock_endpoint_manager_instance.retrieve_by_fields.assert_called_once()

                # Verify workflow was created
                mock_workflow_service_instance.retrieve_or_create_workflow.assert_called_once()

                # Verify endpoint status was updated to DELETED
                mock_endpoint_manager_instance.update_by_fields.assert_called()
                delete_call = None
                for call in mock_endpoint_manager_instance.update_by_fields.call_args_list:
                    if call[0][1].get("status") == EndpointStatusEnum.DELETED:
                        delete_call = call
                        break
                assert delete_call is not None, "Endpoint should be marked as DELETED"

                # Verify workflow status was updated to COMPLETED
                mock_workflow_manager_instance.update_by_fields.assert_called()
                workflow_call = None
                for call in mock_workflow_manager_instance.update_by_fields.call_args_list:
                    if call[0][1].get("status") == WorkflowStatusEnum.COMPLETED:
                        workflow_call = call
                        break
                assert workflow_call is not None, "Workflow should be marked as COMPLETED"

                # Since we're not mocking provider manager, we don't need to check it was called

                # Verify Redis cache was updated
                mock_delete_proxy.assert_called_once_with(endpoint_id)

                # Verify credential service was called to update proxy cache
                mock_credential_service_instance.update_proxy_cache.assert_called_once_with(project_id)

                # Verify notification was sent
                mock_notify_service_instance.send_notification.assert_called_once()


@pytest.mark.asyncio
async def test_delete_regular_model_endpoint_workflow_deletion():
    """Test that regular model endpoints follow the traditional workflow deletion process."""
    # Setup
    endpoint_id = uuid4()
    user_id = uuid4()
    project_id = uuid4()
    model_id = uuid4()
    workflow_id = uuid4()
    cluster_id = uuid4()

    # Create mock objects with actual attributes
    mock_project = MockProject(id=project_id, name="test-project")
    mock_model = MockModel(
        id=model_id,
        provider_type=ModelProviderTypeEnum.HUGGING_FACE,
        icon="test-icon",
        provider_id=uuid4()
    )
    mock_provider = MagicMock()
    mock_provider.icon = "provider-icon"
    
    mock_cluster = MockCluster(cluster_id=cluster_id)
    mock_endpoint = MockEndpoint(
        id=endpoint_id,
        project_id=project_id,
        model_id=model_id,
        model=mock_model,
        project=mock_project,
        name="test-endpoint",
        status=EndpointStatusEnum.RUNNING,
        cluster=mock_cluster,
        namespace="test-namespace"
    )

    # Mock workflow data
    mock_workflow = MagicMock()
    mock_workflow.id = workflow_id
    mock_workflow.status = WorkflowStatusEnum.IN_PROGRESS
    mock_workflow.created_by = user_id

    # Create service with mocked session
    mock_session = MagicMock()
    service = EndpointService(session=mock_session)

    # Mock database operations and services
    with patch("budapp.endpoint_ops.crud.EndpointDataManager") as mock_endpoint_manager, \
         patch("budapp.workflow_ops.crud.WorkflowDataManager") as mock_workflow_manager, \
         patch("budapp.workflow_ops.crud.WorkflowStepDataManager") as mock_workflow_step_manager, \
         patch("budapp.model_ops.crud.ProviderDataManager") as mock_provider_manager, \
         patch("budapp.credential_ops.services.CredentialService") as mock_credential_service, \
         patch.object(service, '_perform_bud_cluster_delete_endpoint_request') as mock_bud_cluster_delete, \
         patch.object(service, 'delete_model_from_proxy_cache') as mock_delete_proxy:
        
        # Configure endpoint manager
        mock_endpoint_manager_instance = AsyncMock()
        mock_endpoint_manager.return_value = mock_endpoint_manager_instance
        mock_endpoint_manager_instance.retrieve_by_fields.return_value = mock_endpoint
        mock_endpoint_manager_instance.update_by_fields.return_value = mock_endpoint

        # Configure workflow manager
        mock_workflow_manager_instance = AsyncMock()
        mock_workflow_manager.return_value = mock_workflow_manager_instance
        mock_workflow_manager_instance.update_by_fields.return_value = mock_workflow

        # Configure provider manager
        mock_provider_manager_instance = AsyncMock()
        mock_provider_manager.return_value = mock_provider_manager_instance
        mock_provider_manager_instance.retrieve_by_fields.return_value = mock_provider

        # Configure workflow step manager
        mock_workflow_step_manager_instance = AsyncMock()
        mock_workflow_step_manager.return_value = mock_workflow_step_manager_instance
        mock_workflow_step_manager_instance.insert_one.return_value = None

        # Configure credential service
        mock_credential_service_instance = AsyncMock()
        mock_credential_service.return_value = mock_credential_service_instance
        mock_credential_service_instance.update_proxy_cache.return_value = None

        # Mock Redis operations
        mock_delete_proxy.return_value = None

        # Mock bud cluster delete response with required fields
        mock_bud_cluster_delete.return_value = {
            "status": "success",
            "steps": [],
            "workflow_id": str(uuid4())
        }

        # Mock workflow creation
        with patch("budapp.workflow_ops.services.WorkflowService") as mock_workflow_service:
            mock_workflow_service_instance = AsyncMock()
            mock_workflow_service.return_value = mock_workflow_service_instance
            mock_workflow_service_instance.retrieve_or_create_workflow.return_value = mock_workflow

            # Execute
            result = await service.delete_endpoint(endpoint_id, user_id)

            # Verify
            assert result.id == workflow_id
            assert result.status == WorkflowStatusEnum.IN_PROGRESS  # Should still be in progress, not completed

            # Verify endpoint was retrieved
            mock_endpoint_manager_instance.retrieve_by_fields.assert_called_once()

            # Verify workflow was created
            mock_workflow_service_instance.retrieve_or_create_workflow.assert_called_once()

            # Verify endpoint status was updated to DELETING (not DELETED)
            mock_endpoint_manager_instance.update_by_fields.assert_called()
            deleting_call = None
            for call in mock_endpoint_manager_instance.update_by_fields.call_args_list:
                if call[0][1].get("status") == EndpointStatusEnum.DELETING:
                    deleting_call = call
                    break
            assert deleting_call is not None, "Endpoint should be marked as DELETING"

            # Verify workflow status was NOT updated to COMPLETED
            workflow_completed_call = None
            for call in mock_workflow_manager_instance.update_by_fields.call_args_list:
                if call[0][1].get("status") == WorkflowStatusEnum.COMPLETED:
                    workflow_completed_call = call
                    break
            assert workflow_completed_call is None, "Workflow should NOT be marked as COMPLETED"

            # Verify bud_cluster delete was called with expected parameters
            mock_bud_cluster_delete.assert_called_once_with(
                cluster_id, mock_endpoint.namespace, user_id, workflow_id
            )


@pytest.mark.asyncio
async def test_delete_cloud_model_endpoint_with_cluster_follows_workflow():
    """Test that cloud model endpoints with clusters follow the traditional workflow process."""
    # Setup
    endpoint_id = uuid4()
    user_id = uuid4()
    project_id = uuid4()
    model_id = uuid4()
    workflow_id = uuid4()
    cluster_id = uuid4()

    # Create mock objects with actual attributes
    mock_project = MockProject(id=project_id, name="test-project")
    mock_model = MockModel(
        id=model_id,
        provider_type=ModelProviderTypeEnum.CLOUD_MODEL,
        icon="test-icon",
        provider_id=None
    )
    mock_cluster = MockCluster(cluster_id=cluster_id)
    mock_endpoint = MockEndpoint(
        id=endpoint_id,
        project_id=project_id,
        model_id=model_id,
        model=mock_model,
        project=mock_project,
        name="test-endpoint",
        status=EndpointStatusEnum.RUNNING,
        cluster=mock_cluster,
        namespace="test-namespace"
    )

    # Mock workflow data
    mock_workflow = MagicMock()
    mock_workflow.id = workflow_id
    mock_workflow.status = WorkflowStatusEnum.IN_PROGRESS
    mock_workflow.created_by = user_id

    # Create service with mocked session
    mock_session = MagicMock()
    service = EndpointService(session=mock_session)

    # Mock database operations and services
    with patch("budapp.endpoint_ops.crud.EndpointDataManager") as mock_endpoint_manager, \
         patch("budapp.workflow_ops.crud.WorkflowDataManager") as mock_workflow_manager, \
         patch("budapp.workflow_ops.crud.WorkflowStepDataManager") as mock_workflow_step_manager, \
         patch("budapp.credential_ops.services.CredentialService") as mock_credential_service, \
         patch.object(service, '_perform_bud_cluster_delete_endpoint_request') as mock_bud_cluster_delete, \
         patch.object(service, 'delete_model_from_proxy_cache') as mock_delete_proxy:
        
        # Configure endpoint manager
        mock_endpoint_manager_instance = AsyncMock()
        mock_endpoint_manager.return_value = mock_endpoint_manager_instance
        mock_endpoint_manager_instance.retrieve_by_fields.return_value = mock_endpoint
        mock_endpoint_manager_instance.update_by_fields.return_value = mock_endpoint

        # Configure workflow manager
        mock_workflow_manager_instance = AsyncMock()
        mock_workflow_manager.return_value = mock_workflow_manager_instance
        mock_workflow_manager_instance.update_by_fields.return_value = mock_workflow

        # Configure workflow step manager
        mock_workflow_step_manager_instance = AsyncMock()
        mock_workflow_step_manager.return_value = mock_workflow_step_manager_instance
        mock_workflow_step_manager_instance.insert_one.return_value = None

        # Configure credential service
        mock_credential_service_instance = AsyncMock()
        mock_credential_service.return_value = mock_credential_service_instance
        mock_credential_service_instance.update_proxy_cache.return_value = None

        # Mock Redis operations
        mock_delete_proxy.return_value = None

        # Mock bud cluster delete response with required fields
        mock_bud_cluster_delete.return_value = {
            "status": "success",
            "steps": [],
            "workflow_id": str(uuid4())
        }

        # Mock workflow creation
        with patch("budapp.workflow_ops.services.WorkflowService") as mock_workflow_service:
            mock_workflow_service_instance = AsyncMock()
            mock_workflow_service.return_value = mock_workflow_service_instance
            mock_workflow_service_instance.retrieve_or_create_workflow.return_value = mock_workflow

            # Execute
            result = await service.delete_endpoint(endpoint_id, user_id)

            # Verify
            assert result.id == workflow_id
            assert result.status == WorkflowStatusEnum.IN_PROGRESS  # Should still be in progress, not completed

            # Verify endpoint was retrieved
            mock_endpoint_manager_instance.retrieve_by_fields.assert_called_once()

            # Verify workflow was created
            mock_workflow_service_instance.retrieve_or_create_workflow.assert_called_once()

            # Verify endpoint status was updated to DELETING (not DELETED)
            mock_endpoint_manager_instance.update_by_fields.assert_called()
            deleting_call = None
            for call in mock_endpoint_manager_instance.update_by_fields.call_args_list:
                if call[0][1].get("status") == EndpointStatusEnum.DELETING:
                    deleting_call = call
                    break
            assert deleting_call is not None, "Endpoint should be marked as DELETING"

            # Verify workflow status was NOT updated to COMPLETED
            workflow_completed_call = None
            for call in mock_workflow_manager_instance.update_by_fields.call_args_list:
                if call[0][1].get("status") == WorkflowStatusEnum.COMPLETED:
                    workflow_completed_call = call
                    break
            assert workflow_completed_call is None, "Workflow should NOT be marked as COMPLETED"

            # Verify bud_cluster delete was called with expected parameters
            mock_bud_cluster_delete.assert_called_once_with(
                cluster_id, mock_endpoint.namespace, user_id, workflow_id
            )