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

"""The endpoint ops services. Contains business logic for endpoint ops."""

from datetime import datetime, timezone
from typing import Dict, List, Tuple
from uuid import UUID, uuid4

import aiohttp
from fastapi import status

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin
from budapp.project_ops.crud import ProjectDataManager
from budapp.project_ops.models import Project as ProjectModel

from ..cluster_ops.crud import ClusterDataManager
from ..cluster_ops.models import Cluster as ClusterModel
from ..commons.config import app_settings
from ..commons.constants import (
    BUD_INTERNAL_WORKFLOW,
    BudServeWorkflowStepEventName,
    EndpointStatusEnum,
    ModelProviderTypeEnum,
    WorkflowStatusEnum,
    WorkflowTypeEnum,
)
from ..commons.exceptions import ClientException
from ..core.schemas import NotificationPayload
from ..model_ops.crud import ProviderDataManager
from ..model_ops.models import Provider as ProviderModel
from ..model_ops.services import ModelServiceUtil
from ..shared.notification_service import BudNotifyService, NotificationBuilder
from ..workflow_ops.crud import WorkflowDataManager, WorkflowStepDataManager
from ..workflow_ops.models import Workflow as WorkflowModel
from ..workflow_ops.models import WorkflowStep as WorkflowStepModel
from ..workflow_ops.schemas import WorkflowUtilCreate
from ..workflow_ops.services import WorkflowService
from .crud import EndpointDataManager
from .models import Endpoint as EndpointModel
from .schemas import EndpointCreate


logger = logging.get_logger(__name__)


class EndpointService(SessionMixin):
    """Endpoint service."""

    async def get_all_endpoints(
        self,
        project_id: UUID,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[EndpointModel], int]:
        """Get all active endpoints."""
        if search:
            # Only include name, exclude other filters
            # Otherwise it will perform global search on all fields
            filters.pop("status", None)

        # Validate project_id
        await ProjectDataManager(self.session).retrieve_by_fields(ProjectModel, {"id": project_id})

        return await EndpointDataManager(self.session).get_all_active_endpoints(
            project_id, offset, limit, filters, order_by, search
        )

    async def delete_endpoint(self, endpoint_id: UUID, current_user_id: UUID) -> WorkflowModel:
        """Delete an endpoint by its ID."""
        db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
            EndpointModel, {"id": endpoint_id}, exclude_fields={"status": EndpointStatusEnum.DELETED}
        )

        if db_endpoint.model.provider_type in [ModelProviderTypeEnum.HUGGING_FACE, ModelProviderTypeEnum.CLOUD_MODEL]:
            db_provider = await ProviderDataManager(self.session).retrieve_by_fields(
                ProviderModel, {"id": db_endpoint.model.provider_id}
            )
            model_icon = db_provider.icon
        else:
            model_icon = db_endpoint.model.icon

        current_step_number = 1

        # Retrieve or create workflow
        workflow_create = WorkflowUtilCreate(
            workflow_type=WorkflowTypeEnum.ENDPOINT_DELETION,
            title=db_endpoint.name,
            total_steps=current_step_number,
            icon=model_icon,
        )
        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id=None, workflow_data=workflow_create, current_user_id=current_user_id
        )
        logger.debug(f"Delete endpoint workflow {db_workflow.id} created")

        try:
            # Perform delete endpoint request to bud_cluster app
            bud_cluster_response = await self._perform_bud_cluster_delete_endpoint_request(
                db_endpoint.cluster.cluster_id, db_endpoint.namespace, current_user_id, db_workflow.id
            )
        except ClientException as e:
            await WorkflowDataManager(self.session).update_by_fields(
                db_workflow, {"status": WorkflowStatusEnum.FAILED}
            )
            raise e

        # Add payload dict to response
        for step in bud_cluster_response["steps"]:
            step["payload"] = {}

        delete_endpoint_workflow_id = bud_cluster_response.get("workflow_id")
        delete_endpoint_events = {
            BudServeWorkflowStepEventName.DELETE_ENDPOINT_EVENTS.value: bud_cluster_response,
            "delete_endpoint_workflow_id": delete_endpoint_workflow_id,
            "endpoint_id": str(db_endpoint.id),
        }

        # Insert step details in db
        db_workflow_step = await WorkflowStepDataManager(self.session).insert_one(
            WorkflowStepModel(
                workflow_id=db_workflow.id,
                step_number=current_step_number,
                data=delete_endpoint_events,
            )
        )
        logger.debug(f"Created workflow step {current_step_number} for workflow {db_workflow.id}")

        # Update progress in workflow
        bud_cluster_response["progress_type"] = BudServeWorkflowStepEventName.DELETE_ENDPOINT_EVENTS.value
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"progress": bud_cluster_response, "current_step": current_step_number}
        )

        # Update endpoint status to deleting
        await EndpointDataManager(self.session).update_by_fields(db_endpoint, {"status": EndpointStatusEnum.DELETING})
        logger.debug(f"Endpoint {db_endpoint.id} status updated to {EndpointStatusEnum.DELETING.value}")

        return db_workflow

    async def _perform_bud_cluster_delete_endpoint_request(
        self, bud_cluster_id: UUID, namespace: str, current_user_id: UUID, workflow_id: UUID
    ) -> Dict:
        """Perform delete endpoint request to bud_cluster app.

        Args:
            bud_cluster_id: The ID of the cluster being served by the endpoint to delete.
            namespace: The namespace of the cluster endpoint to delete.
        """
        delete_endpoint_url = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_cluster_app_id}/method/deployment/delete"
        )

        payload = {
            "cluster_id": str(bud_cluster_id),
            "namespace": namespace,
            "notification_metadata": {
                "name": BUD_INTERNAL_WORKFLOW,
                "subscriber_ids": str(current_user_id),
                "workflow_id": str(workflow_id),
            },
            "source_topic": f"{app_settings.source_topic}",
        }

        logger.debug(f"Performing delete endpoint request to budcluster {payload}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(delete_endpoint_url, json=payload) as response:
                    response_data = await response.json()
                    if response.status != 200:
                        logger.error(f"Failed to delete endpoint: {response.status} {response_data}")
                        raise ClientException("Failed to delete endpoint")

                    logger.debug("Successfully deleted endpoint from budcluster")
                    return response_data
        except Exception as e:
            logger.exception(f"Failed to send delete endpoint request: {e}")
            raise ClientException("Failed to delete endpoint") from e

    async def delete_endpoint_from_notification_event(self, payload: NotificationPayload) -> None:
        """Delete a endpoint in database.

        Args:
            payload: The payload to delete the endpoint with.

        Raises:
            ClientException: If the endpoint already exists.
        """
        logger.debug("Received event for deleting endpoint")

        # Get workflow and steps
        workflow_id = payload.workflow_id
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Define the keys required for endpoint deletion
        keys_of_interest = [
            "endpoint_id",
        ]

        # from workflow steps extract necessary information
        required_data = {}
        for db_workflow_step in db_workflow_steps:
            for key in keys_of_interest:
                if key in db_workflow_step.data:
                    required_data[key] = db_workflow_step.data[key]

        logger.debug("Collected required data from workflow steps")

        # Retrieve endpoint from db
        db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
            EndpointModel, {"id": required_data["endpoint_id"]}, exclude_fields={"status": EndpointStatusEnum.DELETED}
        )
        logger.debug(f"Endpoint retrieved successfully: {db_endpoint.id}")

        # Mark endpoint as deleted
        db_endpoint = await EndpointDataManager(self.session).update_by_fields(
            db_endpoint, {"status": EndpointStatusEnum.DELETED}
        )
        logger.debug(f"Endpoint {db_endpoint.id} marked as deleted")

        # Mark workflow as completed
        await WorkflowDataManager(self.session).update_by_fields(db_workflow, {"status": WorkflowStatusEnum.COMPLETED})
        logger.debug(f"Workflow {db_workflow.id} marked as completed")

    async def create_endpoint_from_notification_event(self, payload: NotificationPayload) -> None:
        """Create an endpoint in database.

        Args:
            payload: The payload to create the endpoint with.
        """
        logger.debug("Received event for creating endpoint")

        # Get namespace and deployment URL from event
        namespace = payload.content.result.get("namespace")
        deployment_url = payload.content.result["result"]["deployment_url"]
        credential_id = payload.content.result.get("credential_id")

        if not namespace or not deployment_url:
            logger.warning("Namespace or deployment URL is missing from event")
            return

        # Get workflow steps
        workflow_id = payload.workflow_id
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Define the keys required for endpoint creation
        keys_of_interest = [
            "model_id",
            "project_id",
            "cluster_id",  # bud_cluster_id
            "endpoint_name",
        ]

        # from workflow steps extract necessary information
        required_data = {}
        for db_workflow_step in db_workflow_steps:
            for key in keys_of_interest:
                if key in db_workflow_step.data:
                    required_data[key] = db_workflow_step.data[key]

        logger.debug("Collected required data from workflow steps")

        # Get cluster id
        db_cluster = await ClusterDataManager(self.session).retrieve_by_fields(
            ClusterModel, {"cluster_id": required_data["cluster_id"]}, missing_ok=True
        )

        if not db_cluster:
            logger.error(f"Cluster with id {required_data['cluster_id']} not found")
            return

        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})

        # Create endpoint in database
        endpoint_data = EndpointCreate(
            model_id=required_data["model_id"],
            project_id=required_data["project_id"],
            cluster_id=db_cluster.id,
            bud_cluster_id=required_data["cluster_id"],
            name=required_data["endpoint_name"],
            url=deployment_url,
            namespace=namespace,
            status=EndpointStatusEnum.RUNNING,
            created_by=db_workflow.created_by,
            status_sync_at=datetime.now(tz=timezone.utc),
            credential_id=credential_id,
        )

        db_endpoint = await EndpointDataManager(self.session).insert_one(
            EndpointModel(**endpoint_data.model_dump(exclude_unset=True, exclude_none=True))
        )
        logger.debug(f"Endpoint created successfully: {db_endpoint.id}")

        # Mark workflow as completed
        logger.debug(f"Marking workflow as completed: {workflow_id}")
        await WorkflowDataManager(self.session).update_by_fields(db_workflow, {"status": WorkflowStatusEnum.COMPLETED})

        # Send notification to workflow creator
        model_icon = await ModelServiceUtil(self.session).get_model_icon(db_endpoint.model)
        notification_request = (
            NotificationBuilder()
            .set_content(title=db_endpoint.name, message="Deployment is Done", icon=model_icon)
            .set_payload(workflow_id=str(db_workflow.id))
            .set_notification_request(subscriber_ids=[str(db_workflow.created_by)])
            .build()
        )
        await BudNotifyService().send_notification(notification_request)

        # Create request to trigger endpoint status update periodic task
        await self._perform_endpoint_status_update_request(
            db_endpoint.bud_cluster_id, db_endpoint.namespace, db_workflow.created_by
        )

        return db_endpoint

    async def _perform_endpoint_status_update_request(
        self, cluster_id: UUID, namespace: str, current_user_id: UUID
    ) -> Dict:
        """Perform update endpoint status request to bud_cluster app.

        Args:
            cluster_id: The ID of the cluster to update.
            namespace: The namespace of the cluster to update.
            current_user_id: The ID of the current user.
        """
        update_cluster_endpoint = f"{app_settings.dapr_base_url}v1.0/invoke/{app_settings.bud_cluster_app_id}/method/deployment/update-deployment-status"

        try:
            payload = {
                "deployment_name": namespace,
                "cluster_id": str(cluster_id),
                "notification_metadata": {
                    "name": BUD_INTERNAL_WORKFLOW,
                    "subscriber_ids": str(current_user_id),
                    "workflow_id": str(uuid4()),
                },
                "source_topic": f"{app_settings.source_topic}",
            }
            logger.debug(
                f"Performing update endpoint status request. payload: {payload}, endpoint: {update_cluster_endpoint}"
            )
            async with aiohttp.ClientSession() as session, session.post(
                update_cluster_endpoint, json=payload
            ) as response:
                response_data = await response.json()
                if response.status != 200:
                    logger.error(f"Failed to update endpoint status: {response.status} {response_data}")
                    raise ClientException(
                        "Failed to update endpoint status", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )

                logger.debug("Successfully updated endpoint status")
                return response_data
        except Exception as e:
            logger.exception(f"Failed to send update endpoint status request: {e}")
            raise ClientException(
                "Failed to update endpoint status", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from e
