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


"""Implements core services and business logic that power the microservices, including key functionality and integrations."""

from datetime import datetime, timezone
from typing import Dict, List, Tuple

from budapp.cluster_ops.crud import ClusterDataManager
from budapp.cluster_ops.models import Cluster as ClusterModel
from budapp.cluster_ops.services import ClusterService
from budapp.commons import logging
from budapp.commons.constants import BudServeWorkflowStepEventName, EndpointStatusEnum, WorkflowStatusEnum
from budapp.commons.db_utils import SessionMixin
from budapp.endpoint_ops.crud import EndpointDataManager
from budapp.endpoint_ops.models import Endpoint as EndpointModel
from budapp.endpoint_ops.schemas import EndpointCreate
from budapp.model_ops.services import LocalModelWorkflowService
from budapp.workflow_ops.crud import WorkflowDataManager, WorkflowStepDataManager
from budapp.workflow_ops.models import Workflow as WorkflowModel
from budapp.workflow_ops.models import WorkflowStep as WorkflowStepModel

from .crud import IconDataManager
from .models import Icon as IconModel
from .schemas import NotificationPayload


logger = logging.get_logger(__name__)


# Notification related business logic


class NotificationService(SessionMixin):
    """Service for managing notifications."""

    async def update_recommended_cluster_events(self, payload: NotificationPayload) -> None:
        """Update the recommended cluster events for a workflow step.

        Args:
            payload: The payload to update the step with.

        Returns:
            None
        """
        await self._update_workflow_step_events(BudServeWorkflowStepEventName.BUD_SIMULATOR_EVENTS.value, payload)

    async def update_model_deployment_events(self, payload: NotificationPayload) -> None:
        """Update the model deployment events for a workflow step.

        Args:
            payload: The payload to update the step with.

        Returns:
            None
        """
        await self._update_workflow_step_events(BudServeWorkflowStepEventName.BUDSERVE_CLUSTER_EVENTS.value, payload)

        # Create endpoint when deployment is completed
        if (
            payload.content.status == "COMPLETED"
            and payload.content.result
            and isinstance(payload.content.result, dict)
            and "result" in payload.content.result
        ):
            await self._create_endpoint(payload)

    async def update_cluster_creation_events(self, payload: NotificationPayload) -> None:
        """Update the cluster creation events for a workflow step.

        Args:
            payload: The payload to update the step with.

        Returns:
            None
        """
        await self._update_workflow_step_events(BudServeWorkflowStepEventName.CREATE_CLUSTER_EVENTS.value, payload)

        # Create cluster in database if node info fetched successfully
        if (
            payload.content.status == "COMPLETED"
            and payload.content.result
            and payload.content.title == "Fetching cluster nodes info successful"
        ):
            await ClusterService(self.session).create_cluster_from_notification_event(payload)

    async def update_model_extraction_events(self, payload: NotificationPayload) -> None:
        """Update the model extraction events for a workflow step.

        Args:
            payload: The payload to update the step with.

        Returns:
            None
        """
        await self._update_workflow_step_events(BudServeWorkflowStepEventName.MODEL_EXTRACTION_EVENTS.value, payload)

        # Create cluster in database if node info fetched successfully
        if (
            payload.content.status == "COMPLETED"
            and payload.content.result
            and payload.content.title == "Model Extraction Results"
        ):
            await LocalModelWorkflowService(self.session).create_model_from_notification_event(payload)

    async def update_model_security_scan_events(self, payload: NotificationPayload) -> None:
        """Update the model security scan events for a workflow step.

        Args:
            payload: The payload to update the step with.

        Returns:
            None
        """
        await self._update_workflow_step_events(
            BudServeWorkflowStepEventName.MODEL_SECURITY_SCAN_EVENTS.value, payload
        )

        # Create cluster in database if node info fetched successfully
        if (
            payload.content.status == "COMPLETED"
            and payload.content.result
            and payload.content.title == "Model Security Scan Results"
        ):
            await LocalModelWorkflowService(self.session).create_scan_result_from_notification_event(payload)

    async def _update_workflow_step_events(self, event_name: str, payload: NotificationPayload) -> None:
        """Update the workflow step events for a workflow step.

        Args:
            event_name: The name of event to update.
            payload: The payload to update the step with.

        Returns:
            None
        """
        # Fetch workflow steps with simulator events
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps_by_data(
            data_key=event_name, workflow_id=payload.workflow_id
        )

        if not db_workflow_steps:
            logger.warning(f"No workflow steps found for workflow {payload.workflow_id}")
            return

        # Get and validate latest step
        latest_step = db_workflow_steps[-1]

        # Update the payload for the event
        updated_data = await self._update_step_data(event_name, latest_step, payload)

        if not updated_data:
            logger.warning(f"No matching event found for {payload.event}")
            return

        # Update the workflow step data
        # refresh sqlalchemy session otherwise the updated data will not be reflected in the session
        self.session.refresh(latest_step)
        db_workflow_step = await WorkflowStepDataManager(self.session).update_by_fields(
            latest_step, {"data": updated_data}
        )
        logger.info(f"Updated workflow step with {event_name} events: {db_workflow_step.id}")

    async def _update_step_data(self, event_name: str, step: WorkflowStepModel, payload: NotificationPayload) -> dict:
        """Update the payload for the event in the step data.

        Args:
            event_name: The name of event to update.
            step: The workflow step to update.
            payload: The payload to update the step with.

        Returns:
            The updated step data or None if the update failed.
        """
        data = step.data
        simulator_events = data.get(event_name, {})
        steps = simulator_events.get("steps", [])

        if not isinstance(steps, list):
            logger.warning("Steps data is not in expected format")
            return None

        updated = False
        for step_data in steps:
            if isinstance(step_data, dict) and step_data.get("id") == payload.event:
                step_data["payload"] = payload.model_dump(exclude_unset=True, mode="json")
                updated = True
                break

        return data if updated else None

    async def _create_endpoint(self, payload: NotificationPayload) -> None:
        """Create an endpoint when deployment is completed.

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
            "created_by",
            "replicas",
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

        # Create endpoint in database
        endpoint_data = EndpointCreate(
            model_id=required_data["model_id"],
            project_id=required_data["project_id"],
            cluster_id=db_cluster.id,
            bud_cluster_id=required_data["cluster_id"],
            name=required_data["endpoint_name"],
            url=deployment_url,
            namespace=namespace,
            replicas=required_data["replicas"],
            status=EndpointStatusEnum.RUNNING,
            created_by=required_data["created_by"],
            status_sync_at=datetime.now(tz=timezone.utc),
            credential_id=credential_id,
        )

        db_endpoint = await EndpointDataManager(self.session).insert_one(
            EndpointModel(**endpoint_data.model_dump(exclude_unset=True, exclude_none=True))
        )
        logger.debug(f"Endpoint created successfully: {db_endpoint.id}")

        # Mark workflow as completed
        logger.debug(f"Marking workflow as completed: {workflow_id}")
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})
        await WorkflowDataManager(self.session).update_by_fields(db_workflow, {"status": WorkflowStatusEnum.COMPLETED})

        return db_endpoint


class IconService(SessionMixin):
    """Service for managing icons."""

    async def get_all_icons(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[IconModel], int]:
        """Get all icon icons."""
        return await IconDataManager(self.session).get_all_icons(offset, limit, filters, order_by, search)
