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

import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from uuid import UUID

import aiohttp
from fastapi import status

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin
from budapp.commons.schemas import BudNotificationMetadata
from budapp.project_ops.crud import ProjectDataManager
from budapp.project_ops.models import Project as ProjectModel

from ..cluster_ops.crud import ClusterDataManager
from ..cluster_ops.models import Cluster as ClusterModel
from ..cluster_ops.services import ClusterService
from ..commons.config import app_settings
from ..commons.constants import (
    APP_ICONS,
    BUD_INTERNAL_WORKFLOW,
    BudServeWorkflowStepEventName,
    EndpointStatusEnum,
    ModelProviderTypeEnum,
    NotificationTypeEnum,
    WorkflowStatusEnum,
    WorkflowTypeEnum,
)
from ..commons.exceptions import ClientException, RedisException
from ..core.schemas import NotificationPayload, NotificationResult
from ..model_ops.crud import ProviderDataManager
from ..model_ops.models import Provider as ProviderModel
from ..model_ops.services import ModelService, ModelServiceUtil
from ..shared.notification_service import BudNotifyService, NotificationBuilder
from ..shared.redis_service import RedisService
from ..workflow_ops.crud import WorkflowDataManager, WorkflowStepDataManager
from ..workflow_ops.models import Workflow as WorkflowModel
from ..workflow_ops.models import WorkflowStep as WorkflowStepModel
from ..workflow_ops.schemas import WorkflowUtilCreate
from ..workflow_ops.services import WorkflowService, WorkflowStepService
from .crud import EndpointDataManager
from .models import Endpoint as EndpointModel
from .schemas import AddWorkerRequest, AddWorkerWorkflowStepData, EndpointCreate, ModelClusterDetail, WorkerInfoFilter


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

        if db_endpoint.status == EndpointStatusEnum.DELETING:
            raise ClientException("Deployment is already deleting")

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
            tag=db_endpoint.project.name,
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

        # Delete endpoint details with pattern “router_config:*:<endpoint_name>“,
        try:
            redis_service = RedisService()
            endpoint_redis_keys = await redis_service.keys(f"router_config:*:{db_endpoint.name}")
            endpoint_redis_keys_count = await redis_service.delete(*endpoint_redis_keys)
            logger.debug(f"Deleted endpoint data from redis: {endpoint_redis_keys_count} keys")
        except (RedisException, Exception) as e:
            logger.error(f"Failed to delete endpoint details from redis: {e}")

        # Mark workflow as completed
        await WorkflowDataManager(self.session).update_by_fields(db_workflow, {"status": WorkflowStatusEnum.COMPLETED})
        logger.debug(f"Workflow {db_workflow.id} marked as completed")

        # Send notification to workflow creator
        model_icon = await ModelServiceUtil(self.session).get_model_icon(db_endpoint.model)
        notification_request = (
            NotificationBuilder()
            .set_content(
                title=db_endpoint.name,
                message="Deployment Deleted",
                icon=model_icon,
                result=NotificationResult(target_id=db_endpoint.project.id, target_type="project").model_dump(
                    exclude_none=True, exclude_unset=True
                ),
            )
            .set_payload(workflow_id=str(db_workflow.id), type=NotificationTypeEnum.DEPLOYMENT_DELETION_SUCCESS.value)
            .set_notification_request(subscriber_ids=[str(db_workflow.created_by)])
            .build()
        )
        await BudNotifyService().send_notification(notification_request)

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
        number_of_nodes = payload.content.result.get("number_of_nodes")
        total_replicas = payload.content.result["deployment_status"]["replicas"]["total"]

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
            "deploy_config",
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

        # Check duplicate name exist in endpoints
        db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
            EndpointModel,
            fields={"name": required_data["endpoint_name"], "project_id": required_data["project_id"]},
            exclude_fields={"status": EndpointStatusEnum.DELETED},
            missing_ok=True,
            case_sensitive=False,
        )
        if db_endpoint:
            logger.error(
                f"An endpoint with name {required_data['endpoint_name']} already exists in project: {required_data['project_id']}"
            )
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
            status=EndpointStatusEnum.RUNNING,
            created_by=db_workflow.created_by,
            status_sync_at=datetime.now(tz=timezone.utc),
            credential_id=credential_id,
            number_of_nodes=number_of_nodes,
            total_replicas=total_replicas,
            deployment_config=required_data["deploy_config"],
        )

        db_endpoint = await EndpointDataManager(self.session).insert_one(
            EndpointModel(**endpoint_data.model_dump(exclude_unset=True, exclude_none=True))
        )
        logger.debug(f"Endpoint created successfully: {db_endpoint.id}")

        # Update endpoint details as next step
        # Update current step number
        current_step_number = db_workflow.current_step + 1
        workflow_current_step = current_step_number

        # Update or create next workflow step
        endpoint_details = {"endpoint_details": payload.content.result}
        db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
            db_workflow.id, current_step_number, endpoint_details
        )
        logger.debug(f"Upsert workflow step {db_workflow_step.id} for storing endpoint details")

        # Mark workflow as completed
        logger.debug(f"Marking workflow as completed: {workflow_id}")
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"status": WorkflowStatusEnum.COMPLETED, "current_step": workflow_current_step}
        )

        # Send notification to workflow creator
        model_icon = await ModelServiceUtil(self.session).get_model_icon(db_endpoint.model)
        notification_request = (
            NotificationBuilder()
            .set_content(
                title=db_endpoint.name,
                message="Deployment is Done",
                icon=model_icon,
                result=NotificationResult(target_id=db_endpoint.id, target_type="endpoint").model_dump(
                    exclude_none=True, exclude_unset=True
                ),
            )
            .set_payload(workflow_id=str(db_workflow.id), type=NotificationTypeEnum.DEPLOYMENT_SUCCESS.value)
            .set_notification_request(subscriber_ids=[str(db_workflow.created_by)])
            .build()
        )
        await BudNotifyService().send_notification(notification_request)

        # Create request to trigger endpoint status update periodic task
        is_cloud_model = db_endpoint.model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL

        await self._perform_endpoint_status_update_request(
            db_endpoint.bud_cluster_id, db_endpoint.namespace, is_cloud_model
        )

        return db_endpoint

    async def _perform_endpoint_status_update_request(
        self, cluster_id: UUID, namespace: str, is_cloud_model: bool
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
                "cloud_model": is_cloud_model,
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

    async def update_endpoint_status_from_notification_event(self, payload: NotificationPayload) -> None:
        """Update an endpoint status in database.

        Args:
            payload: The payload to update the endpoint status with.

        Raises:
            ClientException: If the endpoint already exists.
        """
        logger.debug("Received event for updating endpoint status")

        # Get endpoint from db
        logger.debug(
            f"Retrieving endpoint with bud_cluster_id: {payload.content.result['cluster_id']} and namespace: {payload.content.result['deployment_name']}"
        )
        total_replicas = len(payload.content.result["worker_data_list"])
        logger.debug(f"Number of workers : {total_replicas}")
        db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
            EndpointModel,
            {
                "bud_cluster_id": payload.content.result["cluster_id"],
                "namespace": payload.content.result["deployment_name"],
            },
            exclude_fields={"status": EndpointStatusEnum.DELETED},
        )
        logger.debug(f"Endpoint retrieved successfully: {db_endpoint.id}")

        # Update cluster status
        endpoint_status = await self._get_endpoint_status(payload.content.result["status"])
        db_endpoint = await EndpointDataManager(self.session).update_by_fields(
            db_endpoint, {"status": endpoint_status, "total_replicas": total_replicas}
        )
        logger.debug(
            f"Endpoint {db_endpoint.id} status updated to {endpoint_status} and total replicas to {total_replicas}"
        )

    @staticmethod
    async def _get_endpoint_status(status: str) -> EndpointStatusEnum:
        """Get the endpoint status from the payload.

        Args:
            status: The status to get the endpoint status from.

        Returns:
            EndpointStatusEnum: The endpoint status.
        """
        if status == "ready":
            return EndpointStatusEnum.RUNNING
        elif status == "pending":
            return EndpointStatusEnum.PENDING
        elif status == "ingress_failed" or status == "failed":
            return EndpointStatusEnum.UNHEALTHY
        else:
            logger.error(f"Unknown endpoint status: {status}")
            raise ClientException(f"Unknown endpoint status: {status}")

    async def get_endpoint_workers(
        self,
        endpoint_id: UUID,
        filters: WorkerInfoFilter,
        refresh: bool,
        page: int,
        limit: int,
        order_by: List[str],
        search: bool,
    ) -> dict:
        """Get endpoint workers."""
        db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(EndpointModel, {"id": endpoint_id})
        get_workers_endpoint = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_cluster_app_id}/method/deployment/worker-info"
        )
        filters_dict = filters.model_dump(exclude_none=True)
        payload = {
            "namespace": db_endpoint.namespace,
            "cluster_id": str(db_endpoint.bud_cluster_id),
            "page": page,
            "limit": limit,
            "order_by": order_by or [],
            "search": str(search).lower(),
            "refresh": str(refresh).lower(),
        }
        logger.info(f"Services : payload: {payload}")
        payload.update(filters_dict)
        headers = {
            "accept": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(get_workers_endpoint, params=payload, headers=headers) as response:
                response_data = await response.json()
                if response.status != 200:
                    error_message = response_data.get("message", "Failed to get endpoint workers")
                    logger.error(f"Failed to get endpoint workers: {error_message}")
                    raise ClientException(error_message)

                logger.debug("Successfully retrieved endpoint workers")
                return response_data

    async def get_endpoint_worker_detail(self, endpoint_id: UUID, worker_id: UUID) -> dict:
        """Get endpoint worker detail."""
        _ = await EndpointDataManager(self.session).retrieve_by_fields(EndpointModel, {"id": endpoint_id})
        get_worker_detail_endpoint = f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_cluster_app_id}/method/deployment/worker-info/{worker_id}"
        headers = {
            "accept": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(get_worker_detail_endpoint, headers=headers) as response:
                response_data = await response.json()
                if response.status != 200:
                    error_message = response_data.get("message", "Failed to get endpoint worker detail")
                    logger.error(f"Failed to get endpoint worker detail: {error_message}")
                    raise ClientException(error_message)

                logger.debug("Successfully retrieved endpoint worker detail")
                return response_data

    async def get_model_cluster_detail(self, endpoint_id: UUID) -> ModelClusterDetail:
        """Get model cluster detail."""
        db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(EndpointModel, {"id": endpoint_id})
        model_id = db_endpoint.model_id
        model_detail_json_response = await ModelService(self.session).retrieve_model(model_id)
        model_detail = json.loads(model_detail_json_response.body.decode("utf-8"))
        cluster_id = db_endpoint.cluster_id
        cluster_detail = await ClusterService(self.session).get_cluster_details(cluster_id)
        return ModelClusterDetail(
            id=db_endpoint.id,
            name=db_endpoint.name,
            status=db_endpoint.status,
            model=model_detail["model"],
            cluster=cluster_detail,
            deployment_config=db_endpoint.deployment_config,
        )

    async def delete_worker_from_notification_event(self, payload: NotificationPayload) -> None:
        """Delete a worker in database.

        Args:
            payload: The payload to delete the worker with.

        Raises:
            ClientException: If the worker already exists.
        """
        logger.debug("Received event for deleting worker")

        # Get workflow and steps
        workflow_id = payload.workflow_id
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Define the keys required for worker deletion
        keys_of_interest = [
            "endpoint_id",
            "worker_id",
            "worker_name",
        ]

        # from workflow steps extract necessary information
        required_data = {}
        for db_workflow_step in db_workflow_steps:
            for key in keys_of_interest:
                if key in db_workflow_step.data:
                    required_data[key] = db_workflow_step.data[key]

        logger.debug("Collected required data from workflow steps")

        db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
            EndpointModel, {"id": required_data["endpoint_id"]}, exclude_fields={"status": EndpointStatusEnum.DELETED}
        )
        logger.debug(f"Endpoint retrieved successfully: {db_endpoint.id}")

        # Mark workflow as completed
        await WorkflowDataManager(self.session).update_by_fields(db_workflow, {"status": WorkflowStatusEnum.COMPLETED})
        logger.debug(f"Workflow {db_workflow.id} marked as completed")

        # Send notification to workflow creator
        model_icon = await ModelServiceUtil(self.session).get_model_icon(db_endpoint.model)
        notification_request = (
            NotificationBuilder()
            .set_content(
                title=db_endpoint.name,
                message="Worker Deleted",
                icon=model_icon,
                result=NotificationResult(target_id=db_endpoint.project.id, target_type="project").model_dump(
                    exclude_none=True, exclude_unset=True
                ),
            )
            .set_payload(workflow_id=str(db_workflow.id), type=NotificationTypeEnum.DEPLOYMENT_DELETION_SUCCESS.value)
            .set_notification_request(subscriber_ids=[str(db_workflow.created_by)])
            .build()
        )
        await BudNotifyService().send_notification(notification_request)

    async def _perform_endpoint_worker_delete_request(
        self, worker_id: UUID, workflow_id: UUID, current_user_id: UUID
    ) -> Dict:
        """Perform update endpoint status request to bud_cluster app.

        Args:
            cluster_id: The ID of the cluster to update.
            namespace: The namespace of the cluster to update.
            current_user_id: The ID of the current user.
        """
        delete_worker_endpoint = (
            f"{app_settings.dapr_base_url}v1.0/invoke/{app_settings.bud_cluster_app_id}/method/deployment/worker-info"
        )

        try:
            notification_metadata = BudNotificationMetadata(
                workflow_id=str(workflow_id),
                subscriber_ids=str(current_user_id),
                name=BUD_INTERNAL_WORKFLOW,
            )
            payload = {
                "worker_id": str(worker_id),
                "notification_metadata": notification_metadata.model_dump(mode="json"),
                "source_topic": f"{app_settings.source_topic}",
            }
            logger.debug(
                f"Performing update endpoint status request. payload: {payload}, endpoint: {delete_worker_endpoint}"
            )
            async with aiohttp.ClientSession() as session, session.delete(
                delete_worker_endpoint, json=payload
            ) as response:
                response_data = await response.json()
                if response.status != 200:
                    logger.error(f"Failed to delete worker: {response.status} {response_data}")
                    error_message = response_data.get("message", "Failed to delete worker")
                    raise ClientException(error_message, status_code=response.status)

                logger.debug("Successfully deleted worker")
                return response_data
        except ClientException as e:
            raise e
        except Exception as e:
            logger.exception(f"Failed to send delete worker request: {e}")
            raise ClientException("Failed to delete worker", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR) from e

    async def delete_endpoint_worker(
        self, endpoint_id: UUID, worker_id: UUID, worker_name: str, current_user_id: UUID
    ) -> None:
        """Delete a endpoint worker by its ID."""
        # To check if endpoint exists
        db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
            EndpointModel, {"id": endpoint_id}, exclude_fields={"status": EndpointStatusEnum.DELETED}
        )
        if not db_endpoint:
            logger.error(f"Endpoint with id {endpoint_id} not found")
            raise ClientException(f"Endpoint with id {endpoint_id} not found")

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
            workflow_type=WorkflowTypeEnum.ENDPOINT_WORKER_DELETION,
            title=worker_name,
            total_steps=current_step_number,
            icon=model_icon,
            tag=db_endpoint.project.name,
        )
        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id=None, workflow_data=workflow_create, current_user_id=current_user_id
        )
        logger.debug(f"Delete worker workflow {db_workflow.id} created")

        try:
            # Perform delete endpoint request to bud_cluster app
            bud_cluster_response = await self._perform_endpoint_worker_delete_request(
                worker_id, db_workflow.id, current_user_id
            )
        except ClientException as e:
            await WorkflowDataManager(self.session).update_by_fields(
                db_workflow, {"status": WorkflowStatusEnum.FAILED}
            )
            raise e

        # Add payload dict to response
        for step in bud_cluster_response["steps"]:
            step["payload"] = {}

        delete_worker_workflow_id = bud_cluster_response.get("workflow_id")
        delete_worker_events = {
            BudServeWorkflowStepEventName.DELETE_WORKER_EVENTS.value: bud_cluster_response,
            "delete_worker_workflow_id": delete_worker_workflow_id,
            "endpoint_id": str(db_endpoint.id),
            "worker_id": str(worker_id),
            "worker_name": worker_name,
        }

        # Insert step details in db
        db_workflow_step = await WorkflowStepDataManager(self.session).insert_one(
            WorkflowStepModel(
                workflow_id=db_workflow.id,
                step_number=current_step_number,
                data=delete_worker_events,
            )
        )
        logger.debug(f"Created workflow step {current_step_number} for workflow {db_workflow.id}")

        # Update progress in workflow
        bud_cluster_response["progress_type"] = BudServeWorkflowStepEventName.DELETE_WORKER_EVENTS.value
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"progress": bud_cluster_response, "current_step": current_step_number}
        )

        return db_workflow

    async def add_worker_to_endpoint_workflow(self, current_user_id: UUID, request: AddWorkerRequest) -> WorkflowModel:
        """Add worker to endpoint workflow."""
        # Get request data
        step_number = request.step_number
        workflow_id = request.workflow_id
        workflow_total_steps = request.workflow_total_steps
        trigger_workflow = request.trigger_workflow
        endpoint_id = request.endpoint_id
        additional_concurrency = request.additional_concurrency
        current_step_number = step_number

        # Retrieve or create workflow
        workflow_create = WorkflowUtilCreate(
            workflow_type=WorkflowTypeEnum.ADD_WORKER_TO_ENDPOINT,
            title="Add Worker to Deployment",
            total_steps=workflow_total_steps,
            icon=APP_ICONS["general"]["deployment_mono"],
            tag="Deployment",
        )
        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id, workflow_create, current_user_id
        )

        # Validate endpoint id
        if endpoint_id:
            db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
                EndpointModel, {"id": endpoint_id}, exclude_fields={"status": EndpointStatusEnum.DELETED}
            )

            # Get icon from provider or model
            if db_endpoint.model.provider_type in [
                ModelProviderTypeEnum.CLOUD_MODEL,
                ModelProviderTypeEnum.HUGGING_FACE,
            ]:
                db_provider = await ProviderDataManager(self.session).retrieve_by_fields(
                    ProviderModel, {"id": db_endpoint.model.provider_id}
                )
                model_icon = db_provider.icon
            else:
                model_icon = db_endpoint.model.icon

            # Update title, icon and tag on workflow
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"title": db_endpoint.name, "icon": model_icon, "tag": db_endpoint.project.name},
            )

        # Prepare workflow step data
        workflow_step_data = AddWorkerWorkflowStepData(
            endpoint_id=endpoint_id,
            additional_concurrency=additional_concurrency,
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
            logger.info(f"Workflow {db_workflow.id} step {current_step_number} already exists")

            # Update workflow step data in db
            db_workflow_step = await WorkflowStepDataManager(self.session).update_by_fields(
                db_current_workflow_step,
                {"data": workflow_step_data},
            )
            logger.info(f"Workflow {db_workflow.id} step {current_step_number} updated")
        else:
            logger.info(f"Creating workflow step {current_step_number} for workflow {db_workflow.id}")

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

        # This will ensure workflow step number is updated to the latest step number
        db_workflow = await WorkflowDataManager(self.session).update_by_fields(
            db_workflow,
            {"current_step": workflow_current_step},
        )

        # Perform bud simulation
        if additional_concurrency:
            db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
                {"workflow_id": db_workflow.id}
            )

            # Define the keys required for model security scan
            keys_of_interest = [
                "endpoint_id",
                "additional_concurrency",
            ]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            # Check if all required keys are present
            required_keys = ["endpoint_id", "additional_concurrency"]
            missing_keys = [key for key in required_keys if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data for bud simulation: {', '.join(missing_keys)}")

            # Perform add worker bud simulation
            try:
                await self._perform_add_worker_simulation(
                    current_step_number, required_data, db_workflow, current_user_id
                )
            except ClientException as e:
                raise e

        # Trigger workflow
        if trigger_workflow:
            # query workflow steps again to get latest data
            db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
                {"workflow_id": db_workflow.id}
            )

            # Define the keys required for model extraction
            keys_of_interest = [
                "endpoint_id",
                "additional_concurrency",
                "simulator_id",
                "deploy_config",
            ]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            # Check if all required keys are present
            required_keys = ["endpoint_id", "additional_concurrency", "simulator_id", "deploy_config"]
            missing_keys = [key for key in required_keys if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data for add worker to deployment: {', '.join(missing_keys)}")

            try:
                # Perform add worker deployment
                await self._perform_add_worker_to_deployment(
                    current_step_number, required_data, db_workflow, current_user_id
                )
            except ClientException as e:
                raise e

        return db_workflow

    async def _perform_add_worker_simulation(
        self, current_step_number: int, data: Dict, db_workflow: WorkflowModel, current_user_id: UUID
    ) -> None:
        """Perform bud simulation."""
        db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
            EndpointModel, {"id": data["endpoint_id"]}, exclude_fields={"status": EndpointStatusEnum.DELETED}
        )

        # Create request payload
        deployment_config = db_endpoint.deployment_config
        payload = {
            "pretrained_model_uri": db_endpoint.model.uri,
            "input_tokens": deployment_config["avg_context_length"],
            "output_tokens": deployment_config["avg_sequence_length"],
            "concurrency": data["additional_concurrency"],
            "cluster_id": str(db_endpoint.cluster.cluster_id),
            "notification_metadata": {
                "name": BUD_INTERNAL_WORKFLOW,
                "subscriber_ids": str(current_user_id),
                "workflow_id": str(db_workflow.id),
            },
            "source_topic": f"{app_settings.source_topic}",
        }
        if db_endpoint.model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
            payload["target_ttft"] = 0
            payload["target_throughput_per_user"] = 0
            payload["target_e2e_latency"] = 0
            payload["is_proprietary_model"] = True
        else:
            payload["target_ttft"] = deployment_config["ttft"][0] if deployment_config["ttft"] else None
            payload["target_throughput_per_user"] = (
                deployment_config["per_session_tokens_per_sec"][1]
                if deployment_config["per_session_tokens_per_sec"]
                else None
            )
            payload["target_e2e_latency"] = (
                deployment_config["e2e_latency"][0] if deployment_config["e2e_latency"] else None
            )
            payload["is_proprietary_model"] = False

        # Perform bud simulation request
        bud_simulation_response = await self._perform_bud_simulation_request(payload)

        # Add payload dict to response
        for step in bud_simulation_response["steps"]:
            step["payload"] = {}

        simulator_id = bud_simulation_response.get("workflow_id")

        # NOTE: Dependency with recommended cluster api (GET /clusters/recommended/{workflow_id})
        # NOTE: Replace concurrent_requests with additional_concurrency
        # Required to compare with concurrent_requests in simulator response
        deployment_config["concurrent_requests"] = data["additional_concurrency"]
        bud_simulation_events = {
            "simulator_id": simulator_id,
            BudServeWorkflowStepEventName.BUD_SIMULATOR_EVENTS.value: bud_simulation_response,
            "deploy_config": deployment_config,
            "model_id": str(db_endpoint.model.id),
        }

        # Increment current step number
        current_step_number = current_step_number + 1
        workflow_current_step = current_step_number

        # Update or create next workflow step
        db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
            db_workflow.id, current_step_number, bud_simulation_events
        )
        logger.debug(f"Workflow step created with id {db_workflow_step.id}")

        # Update progress in workflow
        bud_simulation_response["progress_type"] = BudServeWorkflowStepEventName.BUD_SIMULATOR_EVENTS.value
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"progress": bud_simulation_response, "current_step": workflow_current_step}
        )

    async def _perform_bud_simulation_request(self, payload: Dict) -> Dict:
        """Perform model extraction request."""
        bud_simulation_endpoint = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_simulator_app_id}/method/simulator/run"
        )
        logger.debug(f"payload for bud simulation on add worker to endpoint : {payload}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(bud_simulation_endpoint, json=payload) as response:
                    response_data = await response.json()
                    if response.status >= 400:
                        raise ClientException("Unable to perform model extraction")

                    return response_data
        except ClientException as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to perform model extraction request: {e}")
            raise ClientException("Unable to perform model extraction") from e

    async def _perform_add_worker_to_deployment(
        self, current_step_number: int, data: Dict, db_workflow: WorkflowModel, current_user_id: UUID
    ) -> None:
        """Perform add worker to deployment."""
        # Get endpoint
        db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
            EndpointModel, {"id": data["endpoint_id"]}, exclude_fields={"status": EndpointStatusEnum.DELETED}
        )
        deployment_config = db_endpoint.deployment_config

        # Model URI selection as per lite-llm updates
        db_model = db_endpoint.model
        credential_id = db_endpoint.credential_id
        if db_model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
            model_uri = db_model.uri
            model_source = db_model.source
            if model_uri.startswith(f"{model_source}/"):
                model_uri = model_uri.removeprefix(f"{model_source}/")
            deploy_model_uri = model_uri if not credential_id else f"{model_source}/{model_uri}"
        else:
            deploy_model_uri = db_model.local_path

        add_worker_payload = {
            "cluster_id": str(db_endpoint.cluster_id),
            "simulator_id": data["simulator_id"],
            "endpoint_name": db_endpoint.name,
            "model": deploy_model_uri,
            "concurrency": data["additional_concurrency"],
            "input_tokens": deployment_config["avg_context_length"],
            "output_tokens": deployment_config["avg_sequence_length"],
            "target_throughput_per_user": deployment_config["per_session_tokens_per_sec"][1]
            if deployment_config.get("per_session_tokens_per_sec")
            else None,
            "target_ttft": deployment_config["ttft"][0] if deployment_config.get("ttft") else None,
            "target_e2e_latency": deployment_config["e2e_latency"][0]
            if deployment_config.get("e2e_latency")
            else None,
            "credential_id": str(db_endpoint.credential_id) if db_endpoint.credential_id else None,
            "existing_deployment_namespace": db_endpoint.namespace,
            "notification_metadata": {
                "name": BUD_INTERNAL_WORKFLOW,
                "subscriber_ids": str(current_user_id),
                "workflow_id": str(db_workflow.id),
            },
            "source_topic": f"{app_settings.source_topic}",
        }

        # Perform add worker to deployment request
        add_worker_response = await self._perform_add_worker_to_deployment_request(add_worker_payload)

        # Add payload dict to response
        for step in add_worker_response["steps"]:
            step["payload"] = {}

        add_worker_events = {BudServeWorkflowStepEventName.BUDSERVE_CLUSTER_EVENTS.value: add_worker_response}

        current_step_number = current_step_number + 1
        workflow_current_step = current_step_number

        # Update or create next workflow step
        db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
            db_workflow.id, current_step_number, add_worker_events
        )
        logger.debug(f"Workflow step created with id {db_workflow_step.id}")

        # Update progress in workflow
        add_worker_response["progress_type"] = BudServeWorkflowStepEventName.BUDSERVE_CLUSTER_EVENTS.value
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"progress": add_worker_response, "current_step": workflow_current_step}
        )

    @staticmethod
    async def _perform_add_worker_to_deployment_request(payload: Dict) -> Dict:
        """Perform add worker to deployment request."""
        add_worker_endpoint = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_cluster_app_id}/method/deployment"
        )
        logger.debug(f"payload for add worker to deployment : {payload}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(add_worker_endpoint, json=payload) as response:
                    response_data = await response.json()
                    if response.status >= 400:
                        raise ClientException("Unable to perform add worker to deployment")

                    return response_data
        except ClientException as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to perform add worker to deployment request: {e}")
            raise ClientException("Unable to perform add worker to deployment") from e

    async def add_worker_from_notification_event(self, payload: NotificationPayload) -> None:
        """Add worker from notification event."""
        # Get workflow and workflow steps
        workflow_id = payload.workflow_id
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Define the keys required for endpoint creation
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

        # Get endpoint id
        db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
            EndpointModel,
            {"id": required_data["endpoint_id"]},
            exclude_fields={"status": EndpointStatusEnum.DELETED},
            missing_ok=True,
        )

        if not db_endpoint:
            logger.error(f"Endpoint with id {required_data['endpoint_id']} not found")
            return

        # Add concurrency to existing deployment config
        deployment_config = db_endpoint.deployment_config
        logger.debug(f"Existing deployment config: {deployment_config}")

        existing_concurrency = deployment_config["concurrent_requests"]
        additional_concurrency = payload.content.result.get("result", {}).get("concurrency", 0)
        deployment_config["concurrent_requests"] = existing_concurrency + additional_concurrency

        db_endpoint = await EndpointDataManager(self.session).update_by_fields(
            db_endpoint, {"deployment_config": deployment_config}
        )
        logger.debug(f"Updated deployment config: {deployment_config}")

        # Update current step number
        current_step_number = db_workflow.current_step + 1
        workflow_current_step = current_step_number

        execution_status_data = {
            "workflow_execution_status": {
                "status": "success",
                "message": "Deployment successfully updated with additional concurrency",
            },
        }
        # Update or create next workflow step
        db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
            db_workflow.id, current_step_number, execution_status_data
        )
        logger.debug(f"Upsert workflow step {db_workflow_step.id} for storing endpoint details")

        # Mark workflow as completed
        logger.debug(f"Marking workflow as completed: {workflow_id}")
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"status": WorkflowStatusEnum.COMPLETED, "current_step": workflow_current_step}
        )

        # Send notification to workflow creator
        model_icon = await ModelServiceUtil(self.session).get_model_icon(db_endpoint.model)

        notification_request = (
            NotificationBuilder()
            .set_content(
                title=db_endpoint.name,
                message="Worker Added",
                icon=model_icon,
                result=NotificationResult(target_id=db_endpoint.id, target_type="endpoint").model_dump(
                    exclude_none=True, exclude_unset=True
                ),
            )
            .set_payload(workflow_id=str(db_workflow.id), type=NotificationTypeEnum.DEPLOYMENT_SUCCESS.value)
            .set_notification_request(subscriber_ids=[str(db_workflow.created_by)])
            .build()
        )
        await BudNotifyService().send_notification(notification_request)
