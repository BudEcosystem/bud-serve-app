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

from budapp.commons import logging
from budapp.commons.constants import BudServeWorkflowStepEventName
from budapp.commons.db_utils import SessionMixin

from .crud import WorkflowStepDataManager
from .models import WorkflowStep as WorkflowStepModel
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
