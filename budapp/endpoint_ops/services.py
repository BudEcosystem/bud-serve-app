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

from typing import Dict, List, Tuple
from uuid import UUID

import aiohttp

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin
from budapp.project_ops.crud import ProjectDataManager
from budapp.project_ops.models import Project as ProjectModel

from ..commons.config import app_settings
from ..commons.constants import BudServeWorkflowStepEventName, EndpointStatusEnum
from ..commons.exceptions import ClientException
from ..workflow_ops.crud import WorkflowStepDataManager
from ..workflow_ops.models import WorkflowStep as WorkflowStepModel
from ..workflow_ops.services import WorkflowService
from .crud import EndpointDataManager
from .models import Endpoint as EndpointModel


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

    async def delete_endpoint(self, endpoint_id: UUID, current_user_id: UUID) -> None:
        """Delete an endpoint by its ID."""
        db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
            EndpointModel, {"id": endpoint_id, "is_active": True}
        )

        current_step_number = 1

        # Retrieve or create workflow
        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id=None, workflow_total_steps=current_step_number, current_user_id=current_user_id
        )
        logger.debug(f"Delete endpoint workflow {db_workflow.id} created")

        # Perform delete cluster request to bud_cluster app
        bud_cluster_response = await self._perform_bud_cluster_delete_endpoint_request(
            db_endpoint.cluster.cluster_id, db_endpoint.namespace, current_user_id, db_workflow.id
        )

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

        # Update cluster status to deleting
        await EndpointDataManager(self.session).update_by_fields(db_endpoint, {"status": EndpointStatusEnum.DELETING})
        logger.debug(f"Endpoint {db_endpoint.id} status updated to {EndpointStatusEnum.DELETING.value}")

        return db_workflow

    async def _perform_bud_cluster_delete_endpoint_request(
        self, bud_cluster_id: UUID, namespace: str, current_user_id: UUID, workflow_id: UUID
    ) -> Dict:
        """Perform delete cluster request to bud_cluster app.

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
                "name": "bud-notification",
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
