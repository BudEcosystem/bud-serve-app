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
from uuid import UUID, uuid4

import pytest

from budapp.commons.constants import (
    EndpointStatusEnum,
    ModelProviderTypeEnum,
    WorkflowStatusEnum,
)
from budapp.endpoint_ops.models import Endpoint as EndpointModel
from budapp.endpoint_ops.services import EndpointService
from budapp.model_ops.models import Model
from budapp.project_ops.models import Project
from budapp.workflow_ops.models import Workflow


@pytest.mark.asyncio
async def test_delete_cloud_model_endpoint_immediate_deletion():
    """Test that cloud model endpoints without clusters are deleted immediately."""
    # Setup
    endpoint_id = uuid4()
    user_id = uuid4()
    project_id = uuid4()
    model_id = uuid4()
    workflow_id = uuid4()

    # Mock model data
    mock_model = MagicMock()
    mock_model.id = model_id
    mock_model.provider_type = ModelProviderTypeEnum.CLOUD_MODEL
    mock_model.source = "openai"
    mock_model.uri = "gpt-4"
    mock_model.icon = "test-icon"

    # Mock project data
    mock_project = MagicMock()
    mock_project.id = project_id
    mock_project.name = "test-project"
    mock_project.name = "test-project"

    # Mock endpoint data
    mock_endpoint = MagicMock()
    mock_endpoint.id = endpoint_id
    mock_endpoint.project_id = project_id
    mock_endpoint.model_id = model_id
    mock_endpoint.model = mock_model
    mock_endpoint.project = mock_project
    mock_endpoint.name = "test-endpoint"
    mock_endpoint.status = EndpointStatusEnum.RUNNING
    mock_endpoint.cluster = None  # No cluster for cloud model

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
         patch("budapp.credential_ops.services.CredentialService") as mock_credential_service:
        
        # Configure endpoint manager
        mock_endpoint_manager_instance = AsyncMock()
        mock_endpoint_manager.return_value = mock_endpoint_manager_instance
        mock_endpoint_manager_instance.retrieve_by_fields.return_value = mock_endpoint
        mock_endpoint_manager_instance.update_by_fields.return_value = mock_endpoint

        # Configure workflow manager
        mock_workflow_manager_instance = AsyncMock()
        mock_workflow_manager.return_value = mock_workflow_manager_instance
        mock_workflow_manager_instance.update_by_fields.return_value = mock_workflow

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

                # Verify Redis cache was updated
                mock_delete_proxy.assert_called_once_with(endpoint_id)
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

    # Mock model data (non-cloud model)
    mock_model = MagicMock()
    mock_model.id = model_id
    mock_model.provider_type = ModelProviderTypeEnum.HUGGING_FACE  # Not a cloud model
    mock_model.icon = "test-icon"

    # Mock cluster data
    mock_cluster = MagicMock()
    mock_cluster.cluster_id = cluster_id

    # Mock project data
    mock_project = MagicMock()
    mock_project.id = project_id
    mock_project.name = "test-project"

    # Mock endpoint data
    mock_endpoint = MagicMock()
    mock_endpoint.id = endpoint_id
    mock_endpoint.project_id = project_id
    mock_endpoint.model_id = model_id
    mock_endpoint.model = mock_model
    mock_endpoint.project = mock_project
    mock_endpoint.name = "test-endpoint"
    mock_endpoint.status = EndpointStatusEnum.RUNNING
    mock_endpoint.cluster = mock_cluster

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
         patch.object(service, '_perform_bud_cluster_delete_endpoint_request') as mock_bud_cluster_delete:
        
        # Configure endpoint manager
        mock_endpoint_manager_instance = AsyncMock()
        mock_endpoint_manager.return_value = mock_endpoint_manager_instance
        mock_endpoint_manager_instance.retrieve_by_fields.return_value = mock_endpoint
        mock_endpoint_manager_instance.update_by_fields.return_value = mock_endpoint

        # Configure workflow manager
        mock_workflow_manager_instance = AsyncMock()
        mock_workflow_manager.return_value = mock_workflow_manager_instance
        mock_workflow_manager_instance.update_by_fields.return_value = mock_workflow

        # Mock bud cluster delete response
        mock_bud_cluster_delete.return_value = {"status": "success"}

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
            assert workflow_completed_call is None, "Workflow should NOT be marked as COMPLETED for regular models"

            # Verify bud cluster delete was called
            mock_bud_cluster_delete.assert_called_once()


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

    # Mock model data (cloud model)
    mock_model = MagicMock()
    mock_model.id = model_id
    mock_model.provider_type = ModelProviderTypeEnum.CLOUD_MODEL  # Cloud model
    mock_model.icon = "test-icon"

    # Mock cluster data (cloud model with cluster)
    mock_cluster = MagicMock()
    mock_cluster.cluster_id = cluster_id

    # Mock project data
    mock_project = MagicMock()
    mock_project.id = project_id
    mock_project.name = "test-project"

    # Mock endpoint data
    mock_endpoint = MagicMock()
    mock_endpoint.id = endpoint_id
    mock_endpoint.project_id = project_id
    mock_endpoint.model_id = model_id
    mock_endpoint.model = mock_model
    mock_endpoint.project = mock_project
    mock_endpoint.name = "test-endpoint"
    mock_endpoint.status = EndpointStatusEnum.RUNNING
    mock_endpoint.cluster = mock_cluster  # Has cluster

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
         patch.object(service, '_perform_bud_cluster_delete_endpoint_request') as mock_bud_cluster_delete:
        
        # Configure endpoint manager
        mock_endpoint_manager_instance = AsyncMock()
        mock_endpoint_manager.return_value = mock_endpoint_manager_instance
        mock_endpoint_manager_instance.retrieve_by_fields.return_value = mock_endpoint
        mock_endpoint_manager_instance.update_by_fields.return_value = mock_endpoint

        # Configure workflow manager
        mock_workflow_manager_instance = AsyncMock()
        mock_workflow_manager.return_value = mock_workflow_manager_instance
        mock_workflow_manager_instance.update_by_fields.return_value = mock_workflow

        # Mock bud cluster delete response
        mock_bud_cluster_delete.return_value = {"status": "success"}

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
            assert workflow_completed_call is None, "Workflow should NOT be marked as COMPLETED for cloud models with clusters"

            # Verify bud cluster delete was called
            mock_bud_cluster_delete.assert_called_once()