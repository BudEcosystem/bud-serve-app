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

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import status

from budapp.commons import logging
from budapp.commons.constants import (
    WORKFLOW_DELETE_MESSAGES,
    BudServeWorkflowStepEventName,
    VisibilityEnum,
    WorkflowStatusEnum,
)
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.model_ops.crud import (
    CloudModelDataManager,
    ModelDataManager,
    ModelSecurityScanResultDataManager,
    ProviderDataManager,
)
from budapp.model_ops.models import CloudModel, Model
from budapp.model_ops.models import ModelSecurityScanResult as ModelSecurityScanResultModel
from budapp.model_ops.models import Provider as ProviderModel
from budapp.model_ops.schemas import QuantizeModelWorkflowStepData
from budapp.workflow_ops.models import Workflow as WorkflowModel
from budapp.workflow_ops.models import WorkflowStep as WorkflowStepModel

from ..cluster_ops.crud import ClusterDataManager
from ..cluster_ops.models import Cluster as ClusterModel
from ..endpoint_ops.crud import EndpointDataManager
from ..endpoint_ops.models import Endpoint as EndpointModel
from ..project_ops.crud import ProjectDataManager
from ..project_ops.models import Project as ProjectModel
from .crud import WorkflowDataManager, WorkflowStepDataManager
from .schemas import RetrieveWorkflowDataResponse, RetrieveWorkflowStepData, WorkflowUtilCreate


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
            create_cluster_events = required_data.get(BudServeWorkflowStepEventName.CREATE_CLUSTER_EVENTS.value)
            delete_cluster_events = required_data.get(BudServeWorkflowStepEventName.DELETE_CLUSTER_EVENTS.value)
            delete_endpoint_events = required_data.get(BudServeWorkflowStepEventName.DELETE_ENDPOINT_EVENTS.value)
            delete_worker_events = required_data.get(BudServeWorkflowStepEventName.DELETE_WORKER_EVENTS.value)
            model_extraction_events = required_data.get(BudServeWorkflowStepEventName.MODEL_EXTRACTION_EVENTS.value)
            bud_serve_cluster_events = required_data.get(BudServeWorkflowStepEventName.BUDSERVE_CLUSTER_EVENTS.value)
            model_security_scan_events = required_data.get(
                BudServeWorkflowStepEventName.MODEL_SECURITY_SCAN_EVENTS.value
            )
            bud_simulator_events = required_data.get(BudServeWorkflowStepEventName.BUD_SIMULATOR_EVENTS.value)
            security_scan_result_id = required_data.get("security_scan_result_id")
            icon = required_data.get("icon")
            uri = required_data.get("uri")
            author = required_data.get("author")
            tags = required_data.get("tags")
            description = required_data.get("description")
            additional_concurrency = required_data.get("additional_concurrency")
            quantized_model_name = required_data.get("quantized_model_name")

            quantization_config = QuantizeModelWorkflowStepData(
                model_id=model_id,
                quantized_model_name=required_data.get("quantized_model_name"),
                target_type=required_data.get("target_type"),
                target_device=required_data.get("target_device"),
                method=required_data.get("method"),
                weight_config=required_data.get("weight_config"),
                activation_config=required_data.get("activation_config"),
            ) if quantized_model_name else None

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

            db_model_security_scan_result = (
                await ModelSecurityScanResultDataManager(self.session).retrieve_by_fields(
                    ModelSecurityScanResultModel, {"id": UUID(security_scan_result_id)}, missing_ok=True
                )
                if "security_scan_result_id" in required_data
                else None
            )

            db_endpoint = (
                await EndpointDataManager(self.session).retrieve_by_fields(
                    EndpointModel, {"id": UUID(required_data["endpoint_id"])}, missing_ok=True
                )
                if "endpoint_id" in required_data
                else None
            )

            db_project = (
                await ProjectDataManager(self.session).retrieve_by_fields(
                    ProjectModel, {"id": UUID(required_data["project_id"])}, missing_ok=True
                )
                if "project_id" in required_data
                else None
            )

            db_cluster = (
                await ClusterDataManager(self.session).retrieve_by_fields(
                    ClusterModel, {"id": UUID(required_data["cluster_id"])}, missing_ok=True
                )
                if "cluster_id" in required_data
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
                icon=icon if icon else None,
                ingress_url=ingress_url if ingress_url else None,
                create_cluster_events=create_cluster_events if create_cluster_events else None,
                uri=uri if uri else None,
                author=author if author else None,
                tags=tags if tags else None,
                model_extraction_events=model_extraction_events if model_extraction_events else None,
                description=description if description else None,
                security_scan_result_id=security_scan_result_id if security_scan_result_id else None,
                model_security_scan_events=model_security_scan_events if model_security_scan_events else None,
                bud_serve_cluster_events=bud_serve_cluster_events if bud_serve_cluster_events else None,
                security_scan_result=db_model_security_scan_result if db_model_security_scan_result else None,
                delete_cluster_events=delete_cluster_events if delete_cluster_events else None,
                delete_endpoint_events=delete_endpoint_events if delete_endpoint_events else None,
                delete_worker_events=delete_worker_events if delete_worker_events else None,
                endpoint=db_endpoint if db_endpoint else None,
                additional_concurrency=additional_concurrency if additional_concurrency else None,
                bud_simulator_events=bud_simulator_events if bud_simulator_events else None,
                project=db_project if db_project else None,
                cluster=db_cluster if db_cluster else None,
                quantization_config=quantization_config if quantization_config else None
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
                "icon",
                "ingress_url",
                BudServeWorkflowStepEventName.CREATE_CLUSTER_EVENTS.value,
                "cluster_id",
            ],
            "add_local_model": [
                "name",
                "uri",
                "author",
                "tags",
                "icon",
                "provider_type",
                "provider_id",
                BudServeWorkflowStepEventName.MODEL_EXTRACTION_EVENTS.value,
                "model_id",
                "description",
            ],
            "scan_local_model": [
                "model_id",
                "security_scan_result_id",
                "leaderboard",
                BudServeWorkflowStepEventName.MODEL_SECURITY_SCAN_EVENTS.value,
            ],
            "delete_cluster": [
                BudServeWorkflowStepEventName.DELETE_CLUSTER_EVENTS.value,
            ],
            "delete_endpoint": [
                BudServeWorkflowStepEventName.DELETE_ENDPOINT_EVENTS.value,
            ],
            "delete_worker": [
                BudServeWorkflowStepEventName.DELETE_WORKER_EVENTS.value,
            ],
            "add_worker_to_endpoint": [
                BudServeWorkflowStepEventName.BUD_SIMULATOR_EVENTS.value,
                BudServeWorkflowStepEventName.BUDSERVE_CLUSTER_EVENTS.value,
                "endpoint_id",
                "additional_concurrency",
                "cluster_id",
                "project_id",
            ],
            "local_model_quantization": [
                "model_id",
                "target_type",
                "target_device",
                "quantized_model_name",
                "method",
                "weight_config",
                "activation_config"
            ]
        }

        # Combine all lists using set union
        all_keys = set().union(*workflow_keys.values())

        return list(all_keys)

    async def retrieve_or_create_workflow(
        self, workflow_id: Optional[UUID], workflow_data: WorkflowUtilCreate, current_user_id: UUID
    ) -> None:
        """Retrieve or create workflow."""
        workflow_data = workflow_data.model_dump(exclude_none=True, exclude_unset=True)

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
        elif "total_steps" in workflow_data:
            db_workflow = await WorkflowDataManager(self.session).insert_one(
                WorkflowModel(**workflow_data, created_by=current_user_id),
            )
        else:
            raise ClientException("Either workflow_id or total_steps should be provided")

        return db_workflow

    async def mark_workflow_as_completed(self, workflow_id: UUID, current_user_id: UUID) -> WorkflowModel:
        """Mark workflow as completed."""
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(
            WorkflowModel, {"id": workflow_id, "created_by": current_user_id}
        )
        logger.debug(f"Workflow found: {db_workflow.id}")

        # Update status to completed only if workflow is not failed
        if db_workflow.status == WorkflowStatusEnum.FAILED:
            logger.error(f"Workflow {workflow_id} is failed")
            raise ClientException("Workflow is failed")

        return await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"status": WorkflowStatusEnum.COMPLETED}
        )

    async def delete_workflow(self, workflow_id: UUID, current_user_id: UUID) -> None:
        """Delete workflow."""
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(
            WorkflowModel, {"id": workflow_id, "created_by": current_user_id}
        )

        if db_workflow.status != WorkflowStatusEnum.IN_PROGRESS:
            logger.error("Unable to delete failed or completed workflow")
            raise ClientException("Workflow is not in progress state")

        # Define success messages for different workflow types
        success_response = WORKFLOW_DELETE_MESSAGES.get(db_workflow.workflow_type, "Workflow deleted successfully")

        # Delete workflow
        await WorkflowDataManager(self.session).delete_one(db_workflow)

        return success_response

    async def get_all_active_workflows(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[WorkflowModel], int]:
        """Get all active worflows."""
        filters_dict = filters

        # Filter by in progress status
        filters_dict["status"] = WorkflowStatusEnum.IN_PROGRESS
        filters_dict["visibility"] = VisibilityEnum.PUBLIC

        return await WorkflowDataManager(self.session).get_all_workflows(offset, limit, filters_dict, order_by, search)


class WorkflowStepService(SessionMixin):
    """Workflow step service."""

    async def create_or_update_next_workflow_step(
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
