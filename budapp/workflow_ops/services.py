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

"""The workflow ops services. Contains business logic for workflow ops."""

from typing import Any, Dict, List
from uuid import UUID

from fastapi import status

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin
from budapp.model_ops.crud import (
    CloudModelDataManager,
    ModelDataManager,
    ProviderDataManager,
)
from budapp.model_ops.models import CloudModel, Model
from budapp.model_ops.models import Provider as ProviderModel
from budapp.workflow_ops.crud import WorkflowDataManager, WorkflowStepDataManager
from budapp.workflow_ops.models import Workflow as WorkflowModel
from budapp.workflow_ops.models import WorkflowStep as WorkflowStepModel

from .schemas import RetrieveWorkflowDataResponse, RetrieveWorkflowStepData


logger = logging.get_logger(__name__)


class WorkflowService(SessionMixin):
    """Workflow service."""

    async def retrieve_workflow_data(self, workflow_id: UUID) -> RetrieveWorkflowDataResponse:
        """Retrieve workflow data."""
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})

        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Extract required data from workflow steps
        required_data = await self._extract_required_data_from_workflow_steps(db_workflow_steps)

        # Parse workflow step data response
        return await self._parse_workflow_step_data_response(required_data, db_workflow)

    async def _extract_required_data_from_workflow_steps(
        self, db_workflow_steps: List[WorkflowStepModel]
    ) -> Dict[str, Any]:
        """Get required data from workflow steps.

        Args:
            db_workflow_steps: List of workflow steps.

        Returns:
            Dict of required data.
        """
        # Define the keys required data retrieval
        keys_of_interest = await self._get_keys_of_interest()

        # from workflow steps extract necessary information
        required_data = {}
        for db_workflow_step in db_workflow_steps:
            for key in keys_of_interest:
                if key in db_workflow_step.data:
                    required_data[key] = db_workflow_step.data[key]

        return required_data

    async def _parse_workflow_step_data_response(
        self, required_data: Dict[str, Any], db_workflow: WorkflowModel
    ) -> RetrieveWorkflowDataResponse:
        """Parse workflow step data response.

        Args:
            required_data: Dict of required data.
            db_workflow: DB workflow.

        Returns:
            RetrieveWorkflowDataResponse: Retrieve workflow data response.
        """
        if required_data:
            # Collect necessary info according to required data
            provider_type = required_data.get("provider_type")
            provider_id = required_data.get("provider_id")
            cloud_model_id = required_data.get("cloud_model_id")
            model_id = required_data.get("model_id")
            workflow_execution_status = required_data.get("workflow_execution_status")
            leaderboard = required_data.get("leaderboard")
            name = required_data.get("name")
            ingress_url = required_data.get("ingress_url")

            db_provider = (
                await ProviderDataManager(self.session).retrieve_by_fields(
                    ProviderModel, {"id": required_data["provider_id"]}, missing_ok=True
                )
                if "provider_id" in required_data
                else None
            )

            db_cloud_model = (
                await CloudModelDataManager(self.session).retrieve_by_fields(
                    CloudModel, {"id": required_data["cloud_model_id"]}, missing_ok=True
                )
                if "cloud_model_id" in required_data
                else None
            )

            db_model = (
                await ModelDataManager(self.session).retrieve_by_fields(
                    Model, {"id": UUID(required_data["model_id"])}, missing_ok=True
                )
                if "model_id" in required_data
                else None
            )

            workflow_steps = RetrieveWorkflowStepData(
                provider_type=provider_type if provider_type else None,
                provider=db_provider if db_provider else None,
                provider_id=provider_id if provider_id else None,
                cloud_model=db_cloud_model if db_cloud_model else None,
                cloud_model_id=cloud_model_id if cloud_model_id else None,
                model=db_model if db_model else None,
                model_id=model_id if model_id else None,
                workflow_execution_status=workflow_execution_status if workflow_execution_status else None,
                leaderboard=leaderboard if leaderboard else None,
                name=name if name else None,
                ingress_url=ingress_url if ingress_url else None,
            )
        else:
            workflow_steps = RetrieveWorkflowStepData()

        return RetrieveWorkflowDataResponse(
            workflow_id=db_workflow.id,
            status=db_workflow.status,
            current_step=db_workflow.current_step,
            total_steps=db_workflow.total_steps,
            reason=db_workflow.reason,
            workflow_steps=workflow_steps,
            code=status.HTTP_200_OK,
            object="workflow.get",
            message="Workflow data retrieved successfully",
        )

    @staticmethod
    async def _get_keys_of_interest() -> List[str]:
        """Get keys of interest as per different workflows."""
        workflow_keys = {
            "add_cloud_model": [
                "source",
                "name",
                "modality",
                "uri",
                "tags",
                "icon",
                "provider_type",
                "provider_id",
                "cloud_model_id",
                "description",
                "model_id",
                "workflow_execution_status",
                "leaderboard",
            ],
            "create_cluster": [
                "name",
                "ingress_url",
            ],
        }

        # Combine all lists using set union
        all_keys = set().union(*workflow_keys.values())

        return list(all_keys)
