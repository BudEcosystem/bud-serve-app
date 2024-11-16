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

"""The cluster ops services. Contains business logic for model ops."""

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import yaml
from fastapi import UploadFile

from budapp.commons import logging
from budapp.commons.async_utils import check_file_extension
from budapp.commons.constants import WorkflowStatusEnum
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.workflow_ops.crud import WorkflowDataManager, WorkflowStepDataManager
from budapp.workflow_ops.models import Workflow as WorkflowModel
from budapp.workflow_ops.models import WorkflowStep as WorkflowStepModel

from .crud import ClusterDataManager
from .models import Cluster as ClusterModel
from .schemas import (
    ClusterResponse,
    CreateClusterWorkflowRequest,
    CreateClusterWorkflowSteps,
)


logger = logging.get_logger(__name__)


class ClusterService(SessionMixin):
    """Cluster service."""

    async def get_all_active_clusters(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[ClusterResponse], int]:
        """Get all active clusters."""
        filters_dict = filters
        filters_dict["is_active"] = True

        clusters, count = await ClusterDataManager(self.session).get_all_clusters(
            offset, limit, filters_dict, order_by, search
        )
        # Add dummy data and additional fields
        updated_clusters = []
        for cluster in clusters:
            updated_cluster = ClusterResponse(
                id=cluster.id,
                cluster_id=cluster.cluster_id,
                name=cluster.name,
                icon=cluster.icon,
                created_at=cluster.created_at,
                modified_at=cluster.modified_at,
                endpoint_count=12,
                status=cluster.status,
                resources={
                    "available_nodes": cluster.available_workers,
                    "total_nodes": cluster.total_workers,
                    "gpu_count": 4,
                    "cpu_count": 8,
                    "hpu_count": 2,
                },
            )
            updated_clusters.append(updated_cluster)

        return updated_clusters, count

    async def create_cluster_workflow(
        self,
        current_user_id: UUID,
        request: CreateClusterWorkflowRequest,
        configuration_file: UploadFile | None = None,
    ) -> None:
        # Get request data
        workflow_id = request.workflow_id
        workflow_total_steps = request.workflow_total_steps
        step_number = request.step_number
        cluster_name = request.name
        ingress_url = request.ingress_url
        trigger_workflow = request.trigger_workflow

        current_step_number = step_number

        # Retrieve or create workflow
        db_workflow = await self._retrieve_or_create_workflow(workflow_id, workflow_total_steps, current_user_id)

        # Validate the configuration file
        if configuration_file:
            if not await check_file_extension(configuration_file.filename, ["yaml", "yml"]):
                logger.error("Invalid file extension for configuration file")
                raise ClientException("Invalid file extension for configuration file")

            try:
                configuration_yaml = yaml.safe_load(configuration_file.file)
            except yaml.YAMLError as e:
                logger.exception(f"Invalid cluster configuration yaml file found: {e}")
                raise ClientException("Invalid cluster configuration yaml file found")

        # Prepare workflow step data
        workflow_step_data = CreateClusterWorkflowSteps(
            name=cluster_name,
            ingress_url=ingress_url,
            configuration_yaml=configuration_yaml,
        ).model_dump(exclude_none=True, exclude_unset=True, mode="json")

        # Get workflow steps
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": db_workflow.id}
        )

        # For avoiding another db call for record retrieval, storing db object while iterating over db_workflow_steps
        db_current_workflow_step = None

        if db_workflow_steps:
            for db_step in db_workflow_steps:
                # Get current workflow step
                if db_step.step_number == current_step_number:
                    db_current_workflow_step = db_step

        if db_current_workflow_step:
            logger.debug(f"Workflow {db_workflow.id} step {current_step_number} already exists")

            # Update workflow step data in db
            db_workflow_step = await WorkflowStepDataManager(self.session).update_by_fields(
                db_current_workflow_step,
                {"data": workflow_step_data},
            )
            logger.debug(f"Workflow {db_workflow.id} step {current_step_number} updated")
        else:
            logger.debug(f"Creating workflow step {current_step_number} for workflow {db_workflow.id}")

            # Default values are inserted in first step of a workflow
            if not db_workflow_steps:
                workflow_step_data["created_by"] = str(current_user_id)

            # Insert step details in db
            db_workflow_step = await WorkflowStepDataManager(self.session).insert_one(
                WorkflowStepModel(
                    workflow_id=db_workflow.id,
                    step_number=current_step_number,
                    data=workflow_step_data,
                )
            )

        # Update workflow current step as the highest step_number
        db_max_workflow_step_number = max(step.step_number for step in db_workflow_steps) if db_workflow_steps else 0
        workflow_current_step = max(current_step_number, db_max_workflow_step_number)
        logger.info(f"The current step of workflow {db_workflow.id} is {workflow_current_step}")

        # Execute workflow
        # Create next step if workflow is triggered
        if trigger_workflow:
            logger.debug("Workflow triggered")

            # Increment step number of workflow and workflow step
            current_step_number = current_step_number + 1
            workflow_current_step = current_step_number

            # Update or create next workflow step
            db_workflow_step = await self._create_or_update_next_workflow_step(db_workflow.id, current_step_number, {})

            # TODO: Currently querying workflow steps again by ordering steps in ascending order
            # To ensure the latest step update is fetched, Consider excluding it later
            db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
                {"workflow_id": db_workflow.id}
            )

            # Define the keys required for model deployment
            keys_of_interest = [
                "name",
                "ingress_url",
                "configuration_yaml",
                "created_by",
            ]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            # Check if all required keys are present
            required_keys = ["name", "ingress_url", "configuration_yaml", "created_by"]
            missing_keys = [key for key in required_keys if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data: {', '.join(missing_keys)}")

            # Trigger create cluster workflow by step
            await self._execute_create_cluster_workflow(required_data, db_workflow.id)
            logger.debug("Successfully executed create cluster workflow")

        # Update workflow step data in db
        db_workflow = await WorkflowDataManager(self.session).update_by_fields(
            db_workflow,
            {"current_step": workflow_current_step},
        )

        return db_workflow

    async def _retrieve_or_create_workflow(
        self, workflow_id: Optional[UUID], workflow_total_steps: Optional[int], current_user_id: UUID
    ) -> None:
        """Retrieve or create workflow."""
        if workflow_id:
            db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(
                WorkflowModel, {"id": workflow_id}
            )

            if db_workflow.status != WorkflowStatusEnum.IN_PROGRESS:
                logger.error(f"Workflow {workflow_id} is not in progress")
                raise ClientException("Workflow is not in progress")

            if db_workflow.created_by != current_user_id:
                logger.error(f"User {current_user_id} is not the creator of workflow {workflow_id}")
                raise ClientException("User is not authorized to perform this action")
        elif workflow_total_steps:
            db_workflow = await WorkflowDataManager(self.session).insert_one(
                WorkflowModel(total_steps=workflow_total_steps, created_by=current_user_id),
            )
        else:
            raise ClientException("Either workflow_id or workflow_total_steps should be provided")

        return db_workflow

    async def _create_or_update_next_workflow_step(
        self, workflow_id: UUID, step_number: int, data: Dict[str, Any]
    ) -> None:
        """Create or update next workflow step."""
        # Check for workflow step exist or not
        db_workflow_step = await WorkflowStepDataManager(self.session).retrieve_by_fields(
            WorkflowStepModel,
            {"workflow_id": workflow_id, "step_number": step_number},
            missing_ok=True,
        )

        if db_workflow_step:
            db_workflow_step = await WorkflowStepDataManager(self.session).update_by_fields(
                db_workflow_step,
                {
                    "workflow_id": workflow_id,
                    "step_number": step_number,
                    "data": data,
                },
            )
        else:
            # Create a new workflow step
            db_workflow_step = await WorkflowStepDataManager(self.session).insert_one(
                WorkflowStepModel(
                    workflow_id=workflow_id,
                    step_number=step_number,
                    data=data,
                )
            )

        return db_workflow_step

    async def _execute_create_cluster_workflow(self, data: Dict[str, Any], workflow_id: UUID) -> None:
        """Execute create cluster workflow."""
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Latest step
        db_latest_workflow_step = db_workflow_steps[-1]

        # Check for duplicate cluster name
        db_cluster = await ClusterDataManager(self.session).retrieve_by_fields(
            ClusterModel, {"name": data["name"], "is_active": True}, missing_ok=True
        )

        if db_cluster:
            logger.error(f"Cluster {data['name']} already exists")
            raise ClientException(f"Cluster {data['name']} already exists")

        return
