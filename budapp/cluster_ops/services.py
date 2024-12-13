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

import json
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from uuid import UUID

import aiohttp
import yaml
from fastapi import UploadFile

from budapp.commons import logging
from budapp.commons.async_utils import check_file_extension
from budapp.commons.config import app_settings
from budapp.commons.constants import BudServeWorkflowStepEventName, ClusterStatusEnum, WorkflowStatusEnum
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.core.schemas import NotificationPayload
from budapp.endpoint_ops.crud import EndpointDataManager
from budapp.endpoint_ops.models import Endpoint as EndpointModel
from budapp.workflow_ops.crud import WorkflowDataManager, WorkflowStepDataManager
from budapp.workflow_ops.models import Workflow as WorkflowModel
from budapp.workflow_ops.models import WorkflowStep as WorkflowStepModel
from budapp.workflow_ops.services import WorkflowService, WorkflowStepService

from .crud import ClusterDataManager
from .models import Cluster as ClusterModel
from .schemas import (
    ClusterCreate,
    ClusterPaginatedResponse,
    ClusterResourcesInfo,
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
    ) -> Tuple[List[ClusterPaginatedResponse], int]:
        """Get all active clusters."""
        filters_dict = filters
        filters_dict["is_active"] = True

        clusters, count = await ClusterDataManager(self.session).get_all_clusters(
            offset, limit, filters_dict, order_by, search
        )
        # Add dummy data and additional fields
        updated_clusters = []
        for cluster in clusters:
            updated_cluster = ClusterPaginatedResponse(
                id=cluster.id,
                cluster_id=cluster.cluster_id,
                name=cluster.name,
                icon=cluster.icon,
                ingress_url=cluster.ingress_url,
                created_at=cluster.created_at,
                modified_at=cluster.modified_at,
                endpoint_count=12,  # TODO: Add endpoint count
                status=cluster.status,
                gpu_count=cluster.gpu_count,
                cpu_count=cluster.cpu_count,
                hpu_count=cluster.hpu_count,
                cpu_total_workers=cluster.cpu_total_workers,
                cpu_available_workers=cluster.cpu_available_workers,
                gpu_total_workers=cluster.gpu_total_workers,
                gpu_available_workers=cluster.gpu_available_workers,
                hpu_total_workers=cluster.hpu_total_workers,
                hpu_available_workers=cluster.hpu_available_workers,
            )
            updated_clusters.append(updated_cluster)

        return updated_clusters, count

    async def create_cluster_workflow(
        self,
        current_user_id: UUID,
        request: CreateClusterWorkflowRequest,
        configuration_file: UploadFile | None = None,
    ) -> None:
        """Create a cluster workflow.

        Args:
            current_user_id: The current user id.
            request: The request to create the cluster workflow with.
            configuration_file: The configuration file to create the cluster with.

        Raises:
            ClientException: If the cluster already exists.
        """
        # Get request data
        workflow_id = request.workflow_id
        workflow_total_steps = request.workflow_total_steps
        step_number = request.step_number
        cluster_name = request.name
        cluster_icon = request.icon
        ingress_url = request.ingress_url
        trigger_workflow = request.trigger_workflow

        current_step_number = step_number

        # Retrieve or create workflow
        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id, workflow_total_steps, current_user_id
        )

        # Validate the configuration file
        if configuration_file:
            if not await check_file_extension(configuration_file.filename, ["yaml", "yml"]):
                logger.error("Invalid file extension for configuration file")
                raise ClientException("Invalid file extension for configuration file")

            try:
                configuration_yaml = yaml.safe_load(configuration_file.file)
            except yaml.YAMLError as e:
                logger.exception(f"Invalid cluster configuration yaml file found: {e}")
                raise ClientException("Invalid cluster configuration yaml file found") from e

        if cluster_name:
            # Check duplicate cluster name
            db_cluster = await ClusterDataManager(self.session).retrieve_by_fields(
                ClusterModel, {"name": cluster_name, "is_active": True}, missing_ok=True
            )
            if db_cluster:
                raise ClientException("Cluster name already exists")

        # Prepare workflow step data
        workflow_step_data = CreateClusterWorkflowSteps(
            name=cluster_name,
            icon=cluster_icon,
            ingress_url=ingress_url,
            configuration_yaml=configuration_yaml if configuration_file else None,
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
            db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
                db_workflow.id, current_step_number, {}
            )

            # Update workflow step data in db
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"current_step": workflow_current_step},
            )

            # TODO: Currently querying workflow steps again by ordering steps in ascending order
            # To ensure the latest step update is fetched, Consider excluding it later
            db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
                {"workflow_id": db_workflow.id}
            )

            # Define the keys required for cluster creation
            keys_of_interest = [
                "name",
                "icon",
                "ingress_url",
                "configuration_yaml",
            ]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            # Check if all required keys are present
            required_keys = ["name", "icon", "ingress_url", "configuration_yaml"]
            missing_keys = [key for key in required_keys if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data: {', '.join(missing_keys)}")

            # Check duplicate cluster name
            db_cluster = await ClusterDataManager(self.session).retrieve_by_fields(
                ClusterModel, {"name": required_data["name"], "is_active": True}, missing_ok=True
            )
            if db_cluster:
                raise ClientException("Cluster name already exists")

            # Trigger create cluster workflow by step
            await self._execute_create_cluster_workflow(required_data, db_workflow.id, current_user_id)
            logger.debug("Successfully executed create cluster workflow")

            # Increment step number of workflow and workflow step
            current_step_number = current_step_number + 1
            workflow_current_step = current_step_number

            # Create next step for storing success event
            await WorkflowStepService(self.session).create_or_update_next_workflow_step(
                db_workflow.id, current_step_number, {}
            )

        # Update workflow step data in db
        db_workflow = await WorkflowDataManager(self.session).update_by_fields(
            db_workflow,
            {"current_step": workflow_current_step},
        )

        return db_workflow

    async def _execute_create_cluster_workflow(
        self, data: Dict[str, Any], workflow_id: UUID, current_user_id: UUID
    ) -> None:
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

        # Create cluster in bud_cluster app
        bud_cluster_response = await self._perform_create_cluster_request(data, workflow_id, current_user_id)

        # Add payload dict to response
        for step in bud_cluster_response["steps"]:
            step["payload"] = {}

        create_cluster_events = {BudServeWorkflowStepEventName.CREATE_CLUSTER_EVENTS.value: bud_cluster_response}

        # Update workflow step with response
        await WorkflowStepDataManager(self.session).update_by_fields(
            db_latest_workflow_step, {"data": create_cluster_events}
        )

    async def _perform_create_cluster_request(
        self, data: Dict[str, str], workflow_id: UUID, current_user_id: UUID
    ) -> dict:
        """Make async POST request to create cluster to budcluster app.

        Args:
            data (Dict[str, str]): Data to be sent in the request.

        Returns:
            dict: Response from the server.

        Raises:
            aiohttp.ClientError: If the request fails.
        """
        create_cluster_endpoint = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_cluster_app_id}/method/cluster"
        )
        cluster_create_request = {
            "enable_master_node": True,
            "name": data["name"],
            "ingress_url": data["ingress_url"],
            "notification_metadata": {
                "name": "bud-notification",
                "subscriber_ids": str(current_user_id),
                "workflow_id": str(workflow_id),
            },
            "source_topic": f"{app_settings.source_topic}",
        }

        # Create temporary yaml file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as temp_file:
            try:
                # Write configuration yaml to temporary yaml file
                yaml.safe_dump(data["configuration_yaml"], temp_file)

                logger.debug(f"cluster_create_request: {cluster_create_request}")
                # Perform the request as a form data
                async with aiohttp.ClientSession() as session:
                    form = aiohttp.FormData()
                    form.add_field("cluster_create_request", json.dumps(cluster_create_request))

                    # Open the file for reading after writing is complete
                    with open(temp_file.name, "rb") as config_file:
                        form.add_field("configuration", config_file, filename=temp_file.name)
                        try:
                            async with session.post(create_cluster_endpoint, data=form) as response:
                                response_data = await response.json()
                                logger.debug(f"Response from budcluster service: {response_data}")

                                if response.status != 200:
                                    error_message = response_data.get("message", "Failed to create cluster")
                                    logger.error(f"Failed to create cluster with external service: {error_message}")
                                    raise ClientException(error_message)

                                logger.debug("Successfully created cluster with budcluster service")
                                return response_data

                        except ClientException as e:
                            raise e

                        except Exception as e:
                            logger.error(f"Failed to make request to budcluster service: {e}")
                            raise ClientException("Unable to create cluster with external service") from e

            except yaml.YAMLError as e:
                logger.error(f"Failed to process YAML configuration: {e}")
                raise ClientException("Invalid YAML configuration") from e

            except IOError as e:
                logger.error(f"Failed to write temporary file: {e}")
                raise ClientException("Failed to process configuration file") from e

            except ClientException as e:
                raise e

            except Exception as e:
                logger.exception(f"Unexpected error during cluster creation request {e}")
                raise ClientException("Unexpected error during cluster creation") from e

            finally:
                # Delete the temporary file
                temp_file.close()

    async def create_cluster_from_notification_event(self, payload: NotificationPayload) -> None:
        """Create a cluster in database.

        Args:
            payload: The payload to create the cluster with.

        Raises:
            ClientException: If the cluster already exists.
        """
        logger.debug("Received event for creating cluster")

        # Get workflow and steps
        workflow_id = payload.workflow_id
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Get last step
        db_latest_workflow_step = db_workflow_steps[-1]

        # Define the keys required for endpoint creation
        keys_of_interest = [
            "name",
            "icon",
            "ingress_url",
        ]

        # from workflow steps extract necessary information
        required_data = {}
        for db_workflow_step in db_workflow_steps:
            for key in keys_of_interest:
                if key in db_workflow_step.data:
                    required_data[key] = db_workflow_step.data[key]

        logger.debug("Collected required data from workflow steps")

        # Check duplicate cluster name
        db_cluster = await ClusterDataManager(self.session).retrieve_by_fields(
            ClusterModel, {"name": required_data["name"], "is_active": True}, missing_ok=True
        )

        if db_cluster:
            logger.error(f"Cluster {required_data['name']} already exists")
            raise ClientException(f"Cluster {required_data['name']} already exists")

        # Get cluster resources from event
        cluster_resources = await self._calculate_cluster_resources(payload.content.result)
        logger.debug("Cluster resources calculated.")

        # Get bud cluster id from event
        bud_cluster_id = payload.content.result["id"]

        cluster_data = ClusterCreate(
            name=required_data["name"],
            icon=required_data["icon"],
            ingress_url=required_data["ingress_url"],
            created_by=db_workflow.created_by,
            cluster_id=UUID(bud_cluster_id),
            **cluster_resources.model_dump(exclude_unset=True, exclude_none=True),
            status=ClusterStatusEnum.AVAILABLE,
            status_sync_at=datetime.now(tz=timezone.utc),
        )

        # Mark workflow as completed
        logger.debug(f"Updating workflow status: {workflow_id}")

        # Update status for last step
        execution_status = {"status": "success", "message": "Cluster successfully created"}
        try:
            db_cluster = await ClusterDataManager(self.session).insert_one(
                ClusterModel(**cluster_data.model_dump(exclude_unset=True, exclude_none=True))
            )
            logger.debug(f"Cluster created successfully: {db_cluster.id}")
        except Exception as e:
            logger.exception(f"Failed to create cluster: {e}")
            execution_status.update({"status": "error", "message": "Failed to create cluster"})
            workflow_data = {"status": WorkflowStatusEnum.FAILED, "reason": str(e)}
        else:
            workflow_data = {"status": WorkflowStatusEnum.COMPLETED}
        finally:
            execution_status_data = {"workflow_execution_status": execution_status}
            db_workflow_step = await WorkflowStepDataManager(self.session).update_by_fields(
                db_latest_workflow_step, {"data": execution_status_data}
            )
            await WorkflowDataManager(self.session).update_by_fields(db_workflow, workflow_data)

    async def _calculate_cluster_resources(self, data: Dict[str, Any]) -> ClusterResourcesInfo:
        """Calculate the cluster resources.

        Args:
            data: The data to calculate the cluster resources with.

        Returns:
            ClusterResourcesInfo: The cluster resources.
        """
        cpu_count = 0
        gpu_count = 0
        hpu_count = 0
        cpu_total_workers = 0
        gpu_total_workers = 0
        hpu_total_workers = 0

        # Iterate through each node
        for node in data.get("nodes", []):
            # Iterate through devices in each node
            for device in node.get("devices", []):
                # Get the available count and device type
                worker_count = device.get("available_count", 0)
                device_type = device.get("type", "").lower()

                # Increment the appropriate counter
                if device_type == "cpu":
                    cpu_count += 1
                    cpu_total_workers += worker_count
                elif device_type == "gpu":
                    gpu_count += 1
                    gpu_total_workers += worker_count
                elif device_type == "hpu":
                    hpu_count += 1
                    hpu_total_workers += worker_count

        return ClusterResourcesInfo(
            cpu_count=cpu_count,
            gpu_count=gpu_count,
            hpu_count=hpu_count,
            cpu_total_workers=cpu_total_workers,
            cpu_available_workers=cpu_total_workers,
            gpu_total_workers=gpu_total_workers,
            gpu_available_workers=gpu_total_workers,
            hpu_total_workers=hpu_total_workers,
            hpu_available_workers=hpu_total_workers,
        )

    async def edit_cluster(self, cluster_id: UUID, data: Dict[str, Any]) -> ClusterResponse:
        """Edit cloud model by validating and updating specific fields, and saving an uploaded file if provided."""
        # Retrieve existing model
        db_cluster = await ClusterDataManager(self.session).retrieve_by_fields(
            model=ClusterModel, fields={"id": cluster_id}
        )

        if "name" in data:
            duplicate_cluster = await ClusterDataManager(self.session).retrieve_by_fields(
                model=ClusterModel,
                fields={"name": data["name"], "is_active": True},
                exclude_fields={"id": cluster_id},
                missing_ok=True,
            )
            if duplicate_cluster:
                raise ClientException("Cluster name already exists")

        db_cluster = await ClusterDataManager(self.session).update_by_fields(db_cluster, data)

        return db_cluster

    async def get_cluster_details(self, cluster_id: UUID) -> ClusterModel:
        """Retrieve model details by model ID."""
        cluster_details = await ClusterDataManager(self.session).retrieve_by_fields(
            ClusterModel, {"id": cluster_id}, missing_ok=True
        )
        cluster_details = ClusterResponse.model_validate(cluster_details)

        return cluster_details

    async def delete_cluster(self, cluster_id: UUID) -> None:
        """Delete a cluster by its ID.

        Args:
            cluster_id: The ID of the cluster to delete.
        """
        db_cluster = await ClusterDataManager(self.session).retrieve_by_fields(
            ClusterModel, {"id": cluster_id, "is_active": True}
        )

        # Check for active endpoints
        db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
            EndpointModel, {"cluster_id": cluster_id, "is_active": True}, missing_ok=True
        )

        # Raise error if cluster has active endpoints
        if db_endpoint:
            raise ClientException("Cannot delete cluster with active endpoints")

        # Perform delete cluster request to bud_cluster app
        await self._perform_bud_cluster_delete_request(db_cluster.cluster_id)

        # Update cluster status in db
        await ClusterDataManager(self.session).update_by_fields(db_cluster, {"is_active": False})

    async def _perform_bud_cluster_delete_request(self, bud_cluster_id: UUID) -> None:
        """Perform delete cluster request to bud_cluster app.

        Args:
            bud_cluster_id: The ID of the cluster to delete.
        """
        delete_cluster_endpoint = f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_cluster_app_id}/method/cluster/{bud_cluster_id}"

        logger.debug("Performing delete cluster request to budcluster")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(delete_cluster_endpoint) as response:
                    if response.status != 200:
                        logger.error(f"Failed to delete cluster: {response.status} {await response.json()}")
                        raise ClientException("Failed to delete cluster")

            logger.debug("Successfully deleted cluster from budcluster")
        except Exception as e:
            logger.exception(f"Failed to delete cluster: {e}")
            raise ClientException("Failed to delete cluster") from e
