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

"""The model ops services. Contains business logic for model ops."""

import os
import tempfile
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse
from uuid import UUID, uuid4

import aiohttp
import requests
from fastapi import UploadFile, status
from pydantic import HttpUrl

from budapp.commons import logging
from budapp.commons.config import app_settings
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException, MinioException
from budapp.commons.helpers import assign_random_colors_to_names, normalize_value
from budapp.commons.schemas import Tag, Task
from budapp.credential_ops.crud import ProprietaryCredentialDataManager
from budapp.credential_ops.models import ProprietaryCredential as ProprietaryCredentialModel
from budapp.workflow_ops.crud import WorkflowDataManager, WorkflowStepDataManager
from budapp.workflow_ops.models import Workflow as WorkflowModel
from budapp.workflow_ops.models import WorkflowStep as WorkflowStepModel
from budapp.workflow_ops.services import WorkflowService, WorkflowStepService

from ..cluster_ops.crud import ClusterDataManager, ModelClusterRecommendedDataManager
from ..cluster_ops.models import Cluster as ClusterModel
from ..cluster_ops.models import ModelClusterRecommended as ModelClusterRecommendedModel
from ..cluster_ops.schemas import RecommendedClusterRequest
from ..cluster_ops.workflows import ClusterRecommendedSchedulerWorkflows
from ..commons.constants import (
    APP_ICONS,
    BENCHMARK_FIELDS_LABEL_MAPPER,
    BENCHMARK_FIELDS_TYPE_MAPPER,
    BUD_INTERNAL_WORKFLOW,
    HF_AUTHORS_DIR,
    BaseModelRelationEnum,
    BudServeWorkflowStepEventName,
    CloudModelStatusEnum,
    ClusterStatusEnum,
    CredentialTypeEnum,
    EndpointStatusEnum,
    ModelLicenseObjectTypeEnum,
    ModelProviderTypeEnum,
    ModelSecurityScanStatusEnum,
    ModelSourceEnum,
    ModelStatusEnum,
    NotificationTypeEnum,
    ProjectStatusEnum,
    VisibilityEnum,
    WorkflowStatusEnum,
    WorkflowTypeEnum,
)
from ..commons.helpers import validate_huggingface_repo_format
from ..commons.schemas import BudNotificationMetadata
from ..commons.security import RSAHandler
from ..core.crud import ModelTemplateDataManager
from ..core.models import ModelTemplate as ModelTemplateModel
from ..core.schemas import NotificationPayload, NotificationResult
from ..endpoint_ops.crud import EndpointDataManager
from ..endpoint_ops.models import Endpoint as EndpointModel
from ..project_ops.crud import ProjectDataManager
from ..project_ops.models import Project as ProjectModel
from ..shared.minio_store import ModelStore
from ..shared.notification_service import BudNotifyService, NotificationBuilder
from ..workflow_ops.schemas import WorkflowUtilCreate
from .crud import (
    CloudModelDataManager,
    ModelDataManager,
    ModelLicensesDataManager,
    ModelSecurityScanResultDataManager,
    PaperPublishedDataManager,
    ProviderDataManager,
)
from .models import CloudModel, Model, ModelLicenses, PaperPublished
from .models import ModelSecurityScanResult as ModelSecurityScanResultModel
from .models import Provider as ProviderModel
from .schemas import (
    CreateCloudModelWorkflowRequest,
    CreateCloudModelWorkflowResponse,
    CreateCloudModelWorkflowStepData,
    CreateCloudModelWorkflowSteps,
    CreateLocalModelWorkflowRequest,
    CreateLocalModelWorkflowSteps,
    DeploymentTemplateCreate,
    DeploymentWorkflowStepData,
    LeaderboardBenchmark,
    LeaderboardModelInfo,
    LeaderboardTable,
    LocalModelScanRequest,
    LocalModelScanWorkflowStepData,
    ModelArchitectureLLMConfig,
    ModelArchitectureVisionConfig,
    ModelCreate,
    ModelDeploymentRequest,
    ModelDetailSuccessResponse,
    ModelIssue,
    ModelLicensesCreate,
    ModelLicensesModel,
    ModelListResponse,
    ModelResponse,
    ModelSecurityScanResultCreate,
    ModelTree,
    PaperPublishedCreate,
    ScalingSpecification,
    TopLeaderboard,
    TopLeaderboardBenchmark,
)


logger = logging.get_logger(__name__)


class ProviderService(SessionMixin):
    """Provider service."""

    async def get_all_providers(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[ProviderModel], int]:
        """Get all providers."""
        return await ProviderDataManager(self.session).get_all_providers(offset, limit, filters, order_by, search)


class CloudModelWorkflowService(SessionMixin):
    """Model service."""

    async def add_cloud_model_workflow(self, current_user_id: UUID, request: CreateCloudModelWorkflowRequest) -> None:
        """Add a cloud model workflow."""
        # Get request data
        step_number = request.step_number
        workflow_id = request.workflow_id
        workflow_total_steps = request.workflow_total_steps
        provider_type = request.provider_type
        name = request.name
        modality = request.modality
        uri = request.uri
        tags = request.tags
        trigger_workflow = request.trigger_workflow
        provider_id = request.provider_id
        cloud_model_id = request.cloud_model_id
        add_model_modality = request.add_model_modality

        current_step_number = step_number

        # Retrieve or create workflow
        workflow_create = WorkflowUtilCreate(
            workflow_type=WorkflowTypeEnum.CLOUD_MODEL_ONBOARDING,
            title="Cloud Model Onboarding",
            total_steps=workflow_total_steps,
            icon=APP_ICONS["general"]["model_mono"],
            tag="Model Onboarding",
        )
        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id, workflow_create, current_user_id
        )

        # Model source is provider type
        source = None
        if provider_id:
            db_provider = await ProviderDataManager(self.session).retrieve_by_fields(
                ProviderModel, {"id": provider_id}
            )
            source = db_provider.type.value

            # Update icon on workflow
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"icon": db_provider.icon, "title": db_provider.name},
            )

        if provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"title": "Cloud Model"},
            )

        if cloud_model_id:
            db_cloud_model = await CloudModelDataManager(self.session).retrieve_by_fields(
                CloudModel, {"id": cloud_model_id, "status": CloudModelStatusEnum.ACTIVE}
            )

            if db_cloud_model.is_present_in_model:
                raise ClientException("Cloud model is already present in model")

            # Update title on workflow
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"title": db_cloud_model.name},
            )

        if name:
            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model, {"name": name, "status": ModelStatusEnum.ACTIVE}, missing_ok=True
            )
            if db_model:
                raise ClientException("Model name already exists")

            # Update title on workflow
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"title": name},
            )

        # Prepare workflow step data
        workflow_step_data = CreateCloudModelWorkflowSteps(
            provider_type=provider_type,
            source=source if source else None,
            name=name,
            modality=modality,
            uri=uri,
            tags=tags,
            provider_id=provider_id,
            cloud_model_id=cloud_model_id,
            add_model_modality=add_model_modality,
        ).model_dump(exclude_none=True, exclude_unset=True, mode="json")

        # Get workflow steps
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": db_workflow.id}
        )

        # For avoiding another db call for record retrieval, storing db object while iterating over db_workflow_steps
        db_current_workflow_step = None

        if db_workflow_steps:
            await self._validate_duplicate_source_uri_model(source, uri, db_workflow_steps, current_step_number)

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

        # Create next step if workflow is triggered
        if trigger_workflow:
            # Increment step number of workflow and workflow step
            current_step_number = current_step_number + 1
            workflow_current_step = current_step_number

            # Update or create next workflow step
            db_workflow_step = await self._create_or_update_next_workflow_step(db_workflow.id, current_step_number, {})

        # Update workflow step data in db
        db_workflow = await WorkflowDataManager(self.session).update_by_fields(
            db_workflow,
            {"current_step": workflow_current_step},
        )

        # Execute workflow
        if trigger_workflow:
            logger.info("Workflow triggered")

            # TODO: Currently querying workflow steps again by ordering steps in ascending order
            # To ensure the latest step update is fetched, Consider excluding it later
            db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
                {"workflow_id": db_workflow.id}
            )

            # Define the keys required for model deployment
            keys_of_interest = [
                "source",
                "name",
                "modality",
                "uri",
                "tags",
                "provider_type",
                "provider_id",
                "cloud_model_id",
            ]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            # Check if all required keys are present
            required_keys = ["provider_type", "provider_id", "modality", "tags", "name", "source"]
            missing_keys = [key for key in required_keys if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data: {', '.join(missing_keys)}")

            # Check duplicate name exist in model
            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model,
                {"name": required_data["name"], "status": ModelStatusEnum.ACTIVE},
                missing_ok=True,
                case_sensitive=False,
            )
            if db_model:
                raise ClientException("Model name already exists")

            # Trigger deploy model by step
            db_model = await self._execute_add_cloud_model_workflow(required_data, db_workflow.id, current_user_id)
            logger.debug(f"Successfully created model {db_model.id}")

        return db_workflow

    async def _execute_add_cloud_model_workflow(
        self, data: Dict[str, Any], workflow_id: UUID, current_user_id: UUID
    ) -> None:
        """Execute add cloud model workflow."""
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Latest step
        db_latest_workflow_step = db_workflow_steps[-1]

        # if cloud model id is provided, retrieve cloud model
        cloud_model_id = data.get("cloud_model_id")
        db_cloud_model = None
        if cloud_model_id:
            db_cloud_model = await CloudModelDataManager(self.session).retrieve_by_fields(
                CloudModel, {"id": cloud_model_id, "status": CloudModelStatusEnum.ACTIVE}, missing_ok=True
            )
        # Check duplicate name exist in model
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            Model,
            {"name": data["name"], "status": ModelStatusEnum.ACTIVE},
            missing_ok=True,
            case_sensitive=False,
        )
        if db_model:
            raise ClientException("Model name already exists")

        # Prepare model creation data from input
        model_data = await self._prepare_model_data(data, current_user_id, db_cloud_model)

        # Check for duplicate model
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            Model,
            {"uri": model_data.uri, "source": model_data.source, "status": ModelStatusEnum.ACTIVE},
            missing_ok=True,
        )
        if db_model:
            logger.info(f"Model {model_data.uri} with {model_data.source} already exists")
            raise ClientException("Duplicate model uri and source found")

        # Mark workflow completed
        logger.debug(f"Updating workflow status: {workflow_id}")
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})

        execution_status_data = {
            "workflow_execution_status": {
                "status": "success",
                "message": "Model successfully added to the repository",
            },
            "model_id": None,
        }
        try:
            db_model = await ModelDataManager(self.session).insert_one(
                Model(**model_data.model_dump(exclude_unset=True))
            )
        except Exception as e:
            logger.exception(f"Failed to add model to the repository {e}")
            execution_status_data["workflow_execution_status"]["status"] = "error"
            execution_status_data["workflow_execution_status"]["message"] = "Failed to add model to the repository"
            execution_status_data["model_id"] = None
            db_workflow_step = await WorkflowStepDataManager(self.session).update_by_fields(
                db_latest_workflow_step, {"data": execution_status_data}
            )
            await WorkflowDataManager(self.session).update_by_fields(
                db_workflow, {"status": WorkflowStatusEnum.FAILED, "reason": str(e)}
            )
        else:
            execution_status_data["model_id"] = str(db_model.id)
            db_workflow_step = await WorkflowStepDataManager(self.session).update_by_fields(
                db_latest_workflow_step, {"data": execution_status_data}
            )

            if db_cloud_model:
                await CloudModelDataManager(self.session).update_by_fields(
                    db_cloud_model, {"is_present_in_model": True}
                )

            # leaderboard data. TODO: Need to add service for leaderboard
            leaderboard_data = await self._get_leaderboard_data()

            # Add leader board details
            end_step_number = db_workflow_step.step_number + 1

            # Create or update next workflow step
            db_workflow_step = await self._create_or_update_next_workflow_step(
                workflow_id, end_step_number, {"leaderboard": leaderboard_data}
            )

            # Update workflow current step and status
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"current_step": end_step_number, "status": WorkflowStatusEnum.COMPLETED},
            )

            # Trigger recommended cluster scheduler workflow
            await ClusterRecommendedSchedulerWorkflows().__call__(model_id=db_model.id)

            # Send notification to workflow creator
            model_icon = await ModelServiceUtil(self.session).get_model_icon(db_model)
            notification_request = (
                NotificationBuilder()
                .set_content(
                    title=db_model.name,
                    message="Added to Repository",
                    icon=model_icon,
                    result=NotificationResult(target_id=db_model.id, target_type="model").model_dump(
                        exclude_none=True, exclude_unset=True
                    ),
                )
                .set_payload(workflow_id=str(db_workflow.id), type=NotificationTypeEnum.MODEL_ONBOARDING_SUCCESS.value)
                .set_notification_request(subscriber_ids=[str(db_workflow.created_by)])
                .build()
            )
            await BudNotifyService().send_notification(notification_request)

        return db_model

    async def _validate_duplicate_source_uri_model(
        self, source: str, uri: str, db_workflow_steps: List[WorkflowStepModel], current_step_number: int
    ) -> None:
        """Validate duplicate source and uri."""
        db_step_uri = None
        db_step_source = None

        for db_step in db_workflow_steps:
            # Get current workflow step
            if db_step.step_number == current_step_number:
                db_current_workflow_step = db_step

            if "uri" in db_step.data:
                db_step_uri = db_step.data["uri"]
            if "source" in db_step.data:
                db_step_source = db_step.data["source"]

        # Check duplicate endpoint in project
        query_uri = None
        query_source = None
        if uri and db_step_source:
            # If user gives uri but source given in earlier step
            query_uri = uri
            query_source = db_step_source
        elif source and db_step_uri:
            # If user gives source but uri given in earlier step
            query_uri = db_step_uri
            query_source = source
        elif uri and source:
            # if user gives both source and uri
            query_uri = uri
            query_source = source
        elif db_step_uri and db_step_source:
            # if user gives source and uri in earlier step
            query_uri = db_step_uri
            query_source = db_step_source

        if query_uri and query_source:
            # NOTE: A model can only be deployed once in a project
            db_cloud_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model,
                {
                    "uri": query_uri,
                    "source": query_source,
                    "status": ModelStatusEnum.ACTIVE,
                },
                missing_ok=True,
            )
            if db_cloud_model:
                logger.info(f"Model {query_uri} already added with {query_source}")
                raise ClientException("Duplicate model uri and source found")

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

    async def _get_leaderboard_data(self) -> None:
        """Get leaderboard data."""
        return [
            {
                "position": 1,
                "model": "name",
                "IFEval": 13.2,
                "BBH": 13.2,
                "M": 1.2,
            },
            {
                "position": 1,
                "model": "name",
                "IFEval": 13.2,
                "BBH": 13.2,
                "M": 1.2,
            },
            {
                "position": 1,
                "model": "name",
                "IFEval": 13.2,
                "BBH": 13.2,
                "M": 1.2,
            },
            {
                "position": 1,
                "model": "name",
                "IFEval": 13.2,
                "BBH": 13.2,
                "M": 1.2,
            },
        ]

    async def _prepare_model_data(
        self, data: Dict[str, Any], current_user_id: UUID, db_cloud_model: Optional[CloudModel] = None
    ) -> None:
        """Prepare model data."""
        source = data.get("source")
        name = data.get("name")
        modality = data.get("modality")
        uri = data.get("uri")
        tags = data.get("tags")
        provider_type = data.get("provider_type")
        provider_id = data.get("provider_id")

        if db_cloud_model:
            model_data = ModelCreate(
                name=name,
                description=db_cloud_model.description,
                tags=tags,
                tasks=db_cloud_model.tasks,
                author=db_cloud_model.author,
                model_size=db_cloud_model.model_size,
                github_url=db_cloud_model.github_url,
                huggingface_url=db_cloud_model.huggingface_url,
                website_url=db_cloud_model.website_url,
                modality=db_cloud_model.modality,
                source=db_cloud_model.source,
                provider_type=provider_type,
                uri=db_cloud_model.uri,
                created_by=current_user_id,
                provider_id=provider_id,
            )
        else:
            model_data = ModelCreate(
                source=source,
                name=name,
                modality=modality,
                uri=uri,
                tags=tags,
                provider_type=provider_type,
                created_by=current_user_id,
                provider_id=provider_id,
            )

        return model_data

    async def get_cloud_model_workflow(self, workflow_id: UUID) -> CreateCloudModelWorkflowResponse:
        """Get cloud model workflow."""
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})

        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        if not db_workflow_steps:
            return CreateCloudModelWorkflowResponse(
                workflow_id=db_workflow.id,
                status=db_workflow.status,
                current_step=db_workflow.current_step,
                total_steps=db_workflow.total_steps,
                reason=db_workflow.reason,
                workflow_steps=CreateCloudModelWorkflowStepData(),
                code=status.HTTP_200_OK,
                object="cloud_model_workflow.get",
                message="Cloud model workflow retrieved successfully",
            )

        # Define the keys required for model deployment
        keys_of_interest = [
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
        ]

        # from workflow steps extract necessary information
        required_data = {}
        for db_workflow_step in db_workflow_steps:
            for key in keys_of_interest:
                if key in db_workflow_step.data:
                    required_data[key] = db_workflow_step.data[key]

        provider_type = required_data.get("provider_type")
        provider_id = required_data.get("provider_id")
        cloud_model_id = required_data.get("cloud_model_id")
        model_id = required_data.get("model_id")
        workflow_execution_status = required_data.get("workflow_execution_status")
        leaderboard = required_data.get("leaderboard")

        db_provider = (
            await ProviderDataManager(self.session).retrieve_by_fields(
                ProviderModel, {"id": required_data["provider_id"]}, missing_ok=True
            )
            if "provider_id" in required_data
            else None
        )

        db_cloud_model = (
            await CloudModelDataManager(self.session).retrieve_by_fields(
                CloudModel,
                {"id": required_data["cloud_model_id"], "status": CloudModelStatusEnum.ACTIVE},
                missing_ok=True,
            )
            if "cloud_model_id" in required_data
            else None
        )

        db_model = (
            await ModelDataManager(self.session).retrieve_by_fields(
                Model, {"id": UUID(required_data["model_id"]), "status": ModelStatusEnum.ACTIVE}, missing_ok=True
            )
            if "model_id" in required_data
            else None
        )

        return CreateCloudModelWorkflowResponse(
            workflow_id=db_workflow.id,
            status=db_workflow.status,
            current_step=db_workflow.current_step,
            total_steps=db_workflow.total_steps,
            reason=db_workflow.reason,
            workflow_steps=CreateCloudModelWorkflowStepData(
                provider_type=provider_type,
                provider=db_provider,
                provider_id=provider_id,
                cloud_model=db_cloud_model,
                cloud_model_id=cloud_model_id,
                model=db_model,
                model_id=model_id,
                workflow_execution_status=workflow_execution_status,
                leaderboard=leaderboard,
            ),
            code=status.HTTP_200_OK,
            object="cloud_model_workflow.get",
            message="Cloud model workflow retrieved successfully",
        )


class LocalModelWorkflowService(SessionMixin):
    """Local model workflow service."""

    async def add_local_model_workflow(self, current_user_id: UUID, request: CreateLocalModelWorkflowRequest) -> None:
        """Add a local model workflow."""
        # Get request data
        step_number = request.step_number
        workflow_id = request.workflow_id
        workflow_total_steps = request.workflow_total_steps
        provider_type = request.provider_type
        proprietary_credential_id = request.proprietary_credential_id
        name = request.name
        uri = request.uri
        author = request.author
        tags = request.tags
        icon = request.icon
        trigger_workflow = request.trigger_workflow
        add_model_modality = request.add_model_modality

        current_step_number = step_number

        # Retrieve or create workflow
        workflow_create = WorkflowUtilCreate(
            workflow_type=WorkflowTypeEnum.LOCAL_MODEL_ONBOARDING,
            title="Local Model Onboarding",
            total_steps=workflow_total_steps,
            icon=APP_ICONS["general"]["model_mono"],
            tag="Model Onboarding",
        )
        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id, workflow_create, current_user_id
        )

        # Validate proprietary credential id
        if proprietary_credential_id:
            await ProprietaryCredentialDataManager(self.session).retrieve_by_fields(
                ProprietaryCredentialModel, {"id": proprietary_credential_id}
            )

        # Validate model name to be unique
        if name:
            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model, {"name": name, "status": ModelStatusEnum.ACTIVE}, missing_ok=True
            )
            if db_model:
                raise ClientException("Model name should be unique")

            # Update title on workflow
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"title": name},
            )

        # Add provider_id for HuggingFace provider type
        provider_id = None
        if provider_type == ModelProviderTypeEnum.HUGGING_FACE:
            db_provider = await ProviderDataManager(self.session).retrieve_by_fields(
                ProviderModel, {"type": CredentialTypeEnum.HUGGINGFACE}
            )
            provider_id = db_provider.id

            # Update icon, title on workflow
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"icon": db_provider.icon, "title": "Huggingface Model"},
            )
        elif provider_type == ModelProviderTypeEnum.URL:
            # Update icon, title on workflow
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"icon": APP_ICONS["general"]["default_url_model"], "title": "URL"},
            )
        elif provider_type == ModelProviderTypeEnum.DISK:
            # Update icon, title on workflow
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"icon": APP_ICONS["general"]["default_disk_model"], "title": "Disk"},
            )

        # Prepare workflow step data
        workflow_step_data = CreateLocalModelWorkflowSteps(
            provider_type=provider_type,
            proprietary_credential_id=proprietary_credential_id,
            name=name,
            uri=uri,
            author=author,
            tags=tags,
            icon=icon,
            provider_id=provider_id,
            add_model_modality=add_model_modality,
        ).model_dump(exclude_none=True, exclude_unset=True, mode="json")

        # Get workflow steps
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": db_workflow.id}
        )

        # For avoiding another db call for record retrieval, storing db object while iterating over db_workflow_steps
        db_current_workflow_step = None

        # Verify hugging face uri duplication
        await self._verify_provider_type_uri_duplication(provider_type, uri, db_workflow_steps)

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

        if trigger_workflow:
            # query workflow steps again to get latest data
            db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
                {"workflow_id": db_workflow.id}
            )

            # Define the keys required for model extraction
            keys_of_interest = [
                "provider_type",
                "proprietary_credential_id",  # Only required for HuggingFace (Optional)
                "name",
                "uri",
            ]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            # Check if all required keys are present
            required_keys = ["provider_type", "name", "uri"]
            missing_keys = [key for key in required_keys if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data for model extraction: {', '.join(missing_keys)}")

            try:
                # Perform model extraction
                await self._perform_model_extraction(current_step_number, required_data, current_user_id, db_workflow)
            except ClientException as e:
                raise e

        return db_workflow

    async def create_model_from_notification_event(self, payload: NotificationPayload) -> None:
        """Create a local model from notification event."""
        logger.debug("Received event for creating local model")

        # Get workflow and steps
        workflow_id = payload.workflow_id
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Define the keys required for endpoint creation
        keys_of_interest = [
            "provider_type",
            "name",
            "uri",
            "author",
            "tags",
            "icon",
        ]

        # from workflow steps extract necessary information
        required_data = {}
        for db_workflow_step in db_workflow_steps:
            for key in keys_of_interest:
                if key in db_workflow_step.data:
                    required_data[key] = db_workflow_step.data[key]
        logger.debug("Collected required data from workflow steps")

        # Check for model with duplicate name
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            Model,
            {"name": required_data["name"], "status": ModelStatusEnum.ACTIVE},
            missing_ok=True,
            case_sensitive=False,
        )
        if db_model:
            logger.error(f"Unable to create model with name {required_data['name']} as it already exists")
            raise ClientException("Model name should be unique")

        model_info = payload.content.result["model_info"]
        local_path = payload.content.result["local_path"]

        # Extract and finalize tags, tasks and author
        given_tags = required_data.get("tags", [])
        extracted_tags = model_info.get("tags", []) or []
        if extracted_tags:
            # assign random colors to extracted tags
            extracted_tags = assign_random_colors_to_names(extracted_tags)

            # remove duplicate tags
            existing_tag_names = [tag["name"] for tag in given_tags]
            extracted_tags = [tag for tag in extracted_tags if tag["name"] not in existing_tag_names]

        extracted_tags.extend(given_tags)

        extracted_tasks = model_info.get("tasks", [])
        if extracted_tasks:
            extracted_tasks = assign_random_colors_to_names(extracted_tasks)

        extracted_author = required_data.get("author")
        if not extracted_author:
            extracted_author = normalize_value(model_info.get("author", None))

        # Finalize model details
        model_description = normalize_value(model_info.get("description", None))
        model_github_url = normalize_value(model_info.get("github_url", None))
        model_huggingface_url = normalize_value(model_info.get("provider_url", None))
        model_website_url = normalize_value(model_info.get("website_url", None))
        languages = normalize_value(model_info.get("languages", None))
        use_cases = normalize_value(model_info.get("use_cases", None))
        strengths = normalize_value(model_info.get("strengths", None))
        limitations = normalize_value(model_info.get("limitations", None))

        # Architecture
        model_info_architecture = model_info.get("architecture", {})
        if model_info_architecture is not None:
            model_size = normalize_value(model_info_architecture.get("num_params", None))
            model_type = normalize_value(model_info_architecture.get("type", None))
            family = normalize_value(model_info_architecture.get("family", None))
            model_weights_size = normalize_value(model_info_architecture.get("model_weights_size", None))
            kv_cache_size = normalize_value(model_info_architecture.get("kv_cache_size", None))

            # LLM Config
            model_info_text_config = model_info_architecture.get("text_config", {})
            if model_info_text_config is not None:
                text_config = ModelArchitectureLLMConfig(
                    num_layers=normalize_value(model_info_text_config.get("num_layers", None)),
                    hidden_size=normalize_value(model_info_text_config.get("hidden_size", None)),
                    intermediate_size=normalize_value(model_info_text_config.get("intermediate_size", None)),
                    context_length=normalize_value(model_info_text_config.get("context_length", None)),
                    vocab_size=normalize_value(model_info_text_config.get("vocab_size", None)),
                    torch_dtype=normalize_value(model_info_text_config.get("torch_dtype", None)),
                    num_attention_heads=normalize_value(model_info_text_config.get("num_attention_heads", None)),
                    num_key_value_heads=normalize_value(model_info_text_config.get("num_key_value_heads", None)),
                    rope_scaling=normalize_value(model_info_text_config.get("rope_scaling", None)),
                )
            else:
                text_config = None

            # Vision Config
            model_info_vision_config = model_info_architecture.get("vision_config", {})
            if model_info_vision_config is not None:
                vision_config = ModelArchitectureVisionConfig(
                    num_layers=normalize_value(model_info_vision_config.get("num_layers", None)),
                    hidden_size=normalize_value(model_info_vision_config.get("hidden_size", None)),
                    intermediate_size=normalize_value(model_info_vision_config.get("intermediate_size", None)),
                    torch_dtype=normalize_value(model_info_vision_config.get("torch_dtype", None)),
                )
            else:
                vision_config = None
        else:
            model_size = None
            model_type = None
            family = None
            model_weights_size = None
            kv_cache_size = None
            text_config = None
            vision_config = None

        # Get base model relation
        base_model = None
        base_model_relation = None
        model_tree = model_info.get("model_tree", {})

        # Model tree only available for HuggingFace
        if model_tree:
            base_model_relation = await self.get_base_model_relation(model_tree)

            # Sanitize base model
            base_model = model_tree.get("base_model", [])
            if isinstance(base_model, list) and len(base_model) > 0:
                base_model = base_model
            elif isinstance(base_model, str):
                base_model = [base_model]
            base_model = normalize_value(base_model)

            # If base model is the same as the model uri, set base model to None
            if required_data["uri"] == base_model:
                base_model = None

        # Dummy Values
        # TODO: remove this after implementing actual service
        # NOTE: Commented out as it is not required for now, but keeping it here because this structure is integrated in the frontend
        examples = [  # noqa: F841
            {
                "prompt": "Explain the concept of machine learning in simple terms.",
                "prompt_type": "string",
                "response": "Machine learning is like teaching a computer by showing it examples. Just as you learn to recognize cats by seeing many cat pictures, a computer can learn patterns from data to make predictions or decisions.",
                "response_type": "string",
            },
            {
                "prompt": "What are the key differences between AI and human intelligence?",
                "prompt_type": "string",
                "response": "AI excels at processing large amounts of data and finding patterns, while human intelligence shines in creativity, emotional understanding, and adaptability. AI learns from specific data, whereas humans can learn from few examples and apply knowledge across different contexts. Humans have consciousness and self-awareness, which AI currently lacks.",
                "response_type": "string",
            },
        ]
        examples = None  # NOTE: Temporary value, remove after integrating actual examples from model info
        minimum_requirements = {"device_name": "Xenon Dev", "core": 3, "memory": "32 GB", "RAM": "32 GB"}

        # Set provider id and icon
        author_icon = model_info.get("logo_url")
        if author_icon:
            author_icon = LocalModelWorkflowService.save_author_logo(author_icon)
        provider_id = None
        icon = required_data.get("icon") if required_data.get("icon") else author_icon
        if required_data["provider_type"] == ModelProviderTypeEnum.HUGGING_FACE.value:
            # icon is not supported for hugging face models
            # Add provider id for hugging face models to retrieve icon for frontend
            if not icon:
                icon = APP_ICONS["providers"]["default_hugging_face_model"]
            db_provider = await ProviderDataManager(self.session).retrieve_by_fields(
                ProviderModel, {"type": "huggingface"}
            )
            provider_id = db_provider.id

        elif required_data["provider_type"] == ModelProviderTypeEnum.DISK.value:
            if not icon:
                icon = APP_ICONS["general"]["default_disk_model"]
        elif required_data["provider_type"] == ModelProviderTypeEnum.URL.value:
            if not icon:
                icon = APP_ICONS["general"]["default_url_model"]

        model_data = ModelCreate(
            name=required_data["name"],
            description=model_description,
            tags=extracted_tags,
            tasks=extracted_tasks,
            github_url=model_github_url,
            huggingface_url=model_huggingface_url,
            website_url=model_website_url,
            modality=model_info["modality"],
            source=ModelSourceEnum.LOCAL,
            provider_type=required_data["provider_type"],
            uri=required_data["uri"],
            created_by=db_workflow.created_by,
            author=extracted_author,
            provider_id=provider_id,
            local_path=local_path,
            icon=icon,
            model_size=model_size,
            strengths=strengths,
            limitations=limitations,
            languages=languages,
            use_cases=use_cases,
            minimum_requirements=minimum_requirements,
            examples=examples,
            base_model=base_model,
            base_model_relation=base_model_relation,
            model_type=model_type,
            family=family,
            model_weights_size=model_weights_size,
            kv_cache_size=kv_cache_size,
            architecture_text_config=text_config,
            architecture_vision_config=vision_config,
        )

        # Create model
        db_model = await ModelDataManager(self.session).insert_one(Model(**model_data.model_dump(exclude_none=True)))
        logger.debug(f"Model created with id {db_model.id}")

        # Create papers
        extracted_papers = model_info.get("papers", [])
        if extracted_papers:
            db_papers = await self._create_papers_from_model_info(extracted_papers, db_model.id)
            logger.debug(f"Papers created for model {db_model.id}")

        # Create model licenses
        extracted_license = model_info.get("license", {})
        if extracted_license:
            db_model_licenses = await self._create_model_licenses_from_model_info(
                extracted_license, db_model.id, local_path
            )
            logger.debug(f"Model licenses created for model {db_model.id}")

        # Update to workflow step
        workflow_update_data = {
            "model_id": str(db_model.id),
            "tags": extracted_tags,
            "description": model_description,
        }

        current_step_number = db_workflow.current_step + 1
        workflow_current_step = current_step_number

        # Update or create next workflow step
        db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
            workflow_id, current_step_number, workflow_update_data
        )
        logger.debug(f"Workflow step updated {db_workflow_step.id}")

        # Mark workflow as completed
        logger.debug(f"Updating workflow status: {workflow_id}")
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"status": WorkflowStatusEnum.COMPLETED, "current_step": workflow_current_step}
        )

        # Trigger recommended cluster scheduler workflow
        await ClusterRecommendedSchedulerWorkflows().__call__(model_id=db_model.id)

        # Send notification to workflow creator
        model_icon = await ModelServiceUtil(self.session).get_model_icon(db_model)
        notification_request = (
            NotificationBuilder()
            .set_content(
                title=db_model.name,
                message="Added to Repository",
                icon=model_icon,
                result=NotificationResult(target_id=db_model.id, target_type="model").model_dump(
                    exclude_none=True, exclude_unset=True
                ),
            )
            .set_payload(workflow_id=str(db_workflow.id), type=NotificationTypeEnum.MODEL_ONBOARDING_SUCCESS.value)
            .set_notification_request(subscriber_ids=[str(db_workflow.created_by)])
            .build()
        )
        await BudNotifyService().send_notification(notification_request)

    async def _verify_provider_type_uri_duplication(
        self,
        provider_type: ModelProviderTypeEnum,
        uri: str,
        db_workflow_steps: Optional[List[WorkflowStepModel]] = None,
    ) -> None:
        """Verify hugging face uri duplication."""
        db_step_uri = None
        db_step_provider_type = None

        if db_workflow_steps:
            for db_step in db_workflow_steps:
                if "uri" in db_step.data:
                    db_step_uri = db_step.data["uri"]
                if "provider_type" in db_step.data:
                    db_step_provider_type = db_step.data["provider_type"]

        # Check duplicate hugging face uri
        query_uri = None
        query_provider_type = None

        if uri and db_step_provider_type:
            # If user gives uri but provider type given in earlier step
            query_uri = uri
            query_provider_type = db_step_provider_type
        elif provider_type and db_step_uri:
            # If user gives provider type but uri given in earlier step
            query_uri = db_step_uri
            query_provider_type = provider_type.value
        elif uri and provider_type:
            # If user gives both uri and provider type
            query_uri = uri
            query_provider_type = provider_type.value
        elif db_step_uri and db_step_provider_type:
            # If user gives both uri and provider type in earlier step
            query_uri = db_step_uri
            query_provider_type = db_step_provider_type

        if query_uri and query_provider_type and query_provider_type == ModelProviderTypeEnum.HUGGING_FACE.value:
            # Check uri in huggingface uri format
            is_valid_huggingface_uri = validate_huggingface_repo_format(query_uri)
            if not is_valid_huggingface_uri:
                raise ClientException("Invalid huggingface uri format")

        if query_uri and query_provider_type and query_provider_type == ModelProviderTypeEnum.URL.value:
            # Check for valid url
            try:
                HttpUrl(query_uri)
            except ValueError:
                raise ClientException("Invalid url found")

        if query_uri and query_provider_type and query_provider_type == ModelProviderTypeEnum.DISK.value:
            # Check for valid local path
            model_path = os.path.join(app_settings.add_model_dir, query_uri)
            if not os.path.exists(model_path):
                raise ClientException("Given local path does not exist")

        # Check duplicate hugging face uri
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            Model,
            {"uri": query_uri, "provider_type": query_provider_type, "status": ModelStatusEnum.ACTIVE},
            missing_ok=True,
        )
        if db_model:
            raise ClientException(f"Duplicate {query_provider_type} uri found")

    async def _perform_model_extraction(
        self, current_step_number: int, data: Dict, current_user_id: UUID, db_workflow: WorkflowModel
    ) -> None:
        """Perform model extraction."""
        # Perform model extraction request
        model_extraction_response = await self._perform_model_extraction_request(db_workflow.id, data, current_user_id)

        # Add payload dict to response
        for step in model_extraction_response["steps"]:
            step["payload"] = {}

        model_extraction_events = {
            BudServeWorkflowStepEventName.MODEL_EXTRACTION_EVENTS.value: model_extraction_response
        }

        current_step_number = current_step_number + 1
        workflow_current_step = current_step_number

        # Update or create next workflow step
        db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
            db_workflow.id, current_step_number, model_extraction_events
        )
        logger.debug(f"Workflow step created with id {db_workflow_step.id}")

        # Update progress in workflow
        model_extraction_response["progress_type"] = BudServeWorkflowStepEventName.MODEL_EXTRACTION_EVENTS.value
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"progress": model_extraction_response, "current_step": workflow_current_step}
        )

    async def _perform_model_extraction_request(self, workflow_id: UUID, data: Dict, current_user_id: UUID) -> None:
        """Perform model extraction request."""
        model_extraction_endpoint = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_model_app_id}/method/model-info/extract"
        )

        # TODO: get proprietary credential from db
        proprietary_credential_id = data.get("proprietary_credential_id")

        hf_token = None
        if proprietary_credential_id:
            db_proprietary_credential = await ProprietaryCredentialDataManager(self.session).retrieve_by_fields(
                ProprietaryCredentialModel, {"id": proprietary_credential_id}
            )
            hf_token = db_proprietary_credential.other_provider_creds.get("api_key")

            try:
                hf_token = await RSAHandler().decrypt(hf_token)
            except Exception as e:
                logger.error(f"Failed to decrypt token: {e}")
                raise ClientException("Invalid credential found while adding model") from e

        model_extraction_request = {
            "model_name": data["name"],
            "model_uri": data["uri"],
            "provider_type": data["provider_type"],
            "hf_token": hf_token,
            "notification_metadata": {
                "name": BUD_INTERNAL_WORKFLOW,
                "subscriber_ids": str(current_user_id),
                "workflow_id": str(workflow_id),
            },
            "source_topic": f"{app_settings.source_topic}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(model_extraction_endpoint, json=model_extraction_request) as response:
                    response_data = await response.json()
                    if response.status >= 400:
                        raise ClientException("Unable to perform model extraction")

                    return response_data
        except ClientException as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to perform model extraction request: {e}")
            raise ClientException("Unable to perform model extraction") from e

    async def _create_papers_from_model_info(
        self, extracted_papers: list[dict], model_id: UUID
    ) -> List[PaperPublished]:
        """Create papers from model info."""
        # Store papers here for bulk insert
        paper_models = []

        for extracted_paper in extracted_papers:
            paper_title = normalize_value(extracted_paper.get("title", None))
            paper_authors = normalize_value(extracted_paper.get("authors", None))
            paper_url = normalize_value(extracted_paper.get("url", None))

            # Only add paper if it has title, authors or url
            if any([paper_title, paper_authors, paper_url]):
                paper_data = PaperPublishedCreate(
                    title=paper_title,
                    authors=paper_authors,
                    url=paper_url,
                    model_id=model_id,
                )
                paper_models.append(PaperPublished(**paper_data.model_dump(exclude_none=True)))

        # Insert papers in db
        return await PaperPublishedDataManager(self.session).insert_all(paper_models)

    async def _create_model_licenses_from_model_info(
        self, extracted_license: dict, model_id: UUID, local_path: str
    ) -> List[ModelLicenses]:
        """Create model licenses from model info."""
        license_name = normalize_value(extracted_license.get("name", "license"))
        license_url = normalize_value(extracted_license.get("url"))
        license_faqs = normalize_value(extracted_license.get("faqs", []))
        license_type = normalize_value(extracted_license.get("type"))
        license_description = normalize_value(extracted_license.get("description"))
        license_suitability = normalize_value(extracted_license.get("suitability"))
        updated_license_faqs = []
        if license_faqs:
            for faq in license_faqs:
                faq_description = " ".join(faq.get("reason", [])).strip()
                impact = faq.get("impact", "")
                if impact == "POSITIVE":
                    answer = "YES"
                else:
                    answer = "NO"
                updated_license_faqs.append(
                    {
                        "question": faq.get("question"),
                        "description": faq_description,
                        "answer": answer,
                    }
                )

        license_data = ModelLicensesCreate(
            name=license_name,
            url=license_url,
            faqs=updated_license_faqs if updated_license_faqs else None,
            model_id=model_id,
            license_type=license_type,
            description=license_description,
            suitability=license_suitability,
        )
        return await ModelLicensesDataManager(self.session).insert_one(
            ModelLicenses(**license_data.model_dump(exclude_none=True))
        )

    @staticmethod
    async def get_base_model_relation(model_tree: dict) -> Optional[BaseModelRelationEnum]:
        """Get base model relation.

        Args:
            model_tree (dict): Model tree.

        Returns:
            Optional[BaseModelRelationEnum]: Base model relation.
        """
        if model_tree.get("is_finetune"):
            return BaseModelRelationEnum.FINETUNE
        elif model_tree.get("is_adapter"):
            return BaseModelRelationEnum.ADAPTER
        elif model_tree.get("is_quantization"):
            return BaseModelRelationEnum.QUANTIZED
        elif model_tree.get("is_merge"):
            return BaseModelRelationEnum.MERGE
        else:
            return None

    async def scan_local_model_workflow(self, current_user_id: UUID, request: LocalModelScanRequest) -> WorkflowModel:
        """Scan a local model."""
        # Get request data
        step_number = request.step_number
        workflow_id = request.workflow_id
        workflow_total_steps = request.workflow_total_steps
        model_id = request.model_id
        trigger_workflow = request.trigger_workflow

        current_step_number = step_number

        # Retrieve or create workflow
        workflow_create = WorkflowUtilCreate(
            workflow_type=WorkflowTypeEnum.MODEL_SECURITY_SCAN,
            title="Model Security Scan",
            total_steps=workflow_total_steps,
            icon=APP_ICONS["general"]["model_mono"],
            tag="Model Repository",
        )
        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id, workflow_create, current_user_id
        )

        # Validate model id
        if model_id:
            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model, {"id": model_id, "status": ModelStatusEnum.ACTIVE}
            )
            if db_model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
                raise ClientException("Security scan is only supported for local models")

            # Update icon on workflow
            if db_model.provider_type == ModelProviderTypeEnum.HUGGING_FACE:
                db_provider = await ProviderDataManager(self.session).retrieve_by_fields(
                    ProviderModel, {"id": db_model.provider_id}
                )
                model_icon = db_provider.icon
            else:
                model_icon = db_model.icon

            # Update title, icon on workflow
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"title": db_model.name, "icon": model_icon},
            )

        # Prepare workflow step data
        workflow_step_data = LocalModelScanWorkflowStepData(
            model_id=model_id,
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

        if trigger_workflow:
            # query workflow steps again to get latest data
            db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
                {"workflow_id": db_workflow.id}
            )

            # Define the keys required for model security scan
            keys_of_interest = [
                "model_id",
            ]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            # Check if all required keys are present
            required_keys = ["model_id"]
            missing_keys = [key for key in required_keys if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data for model security scan: {', '.join(missing_keys)}")

            try:
                # Perform model security scan
                await self._perform_model_security_scan(current_step_number, required_data, db_workflow)
            except ClientException as e:
                # NOTE: Update workflow status to failed only for model security scan workflow. if micro-service fails,
                # For remaining workflows, if microservice fails, the workflow steps won't be created from backend
                # According to ui model_id and trigger workflow required in request body
                # But Orchestration perspective, the workflow steps can also provided separately in request body
                # So, we need to update the workflow status to failed only for model security scan workflow
                db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                    db_workflow,
                    {"status": WorkflowStatusEnum.FAILED},
                )
                logger.debug("Workflow updated with latest step")
                raise e

        return db_workflow

    async def _perform_model_security_scan(
        self, current_step_number: int, data: Dict, db_workflow: WorkflowModel
    ) -> None:
        """Perform model security scan."""
        # Retrieve workflow step
        db_workflow_step = await WorkflowDataManager(self.session).retrieve_by_fields(
            WorkflowStepModel, {"workflow_id": db_workflow.id, "step_number": current_step_number}
        )

        # Perform model security scan request
        current_user_id = db_workflow_step.workflow.created_by
        model_security_scan_response = await self._perform_model_security_scan_request(
            db_workflow.id, data, current_user_id
        )

        # Add payload dict to response
        for step in model_security_scan_response["steps"]:
            step["payload"] = {}

        # Include model security scan response in current step data
        data[BudServeWorkflowStepEventName.MODEL_SECURITY_SCAN_EVENTS.value] = model_security_scan_response

        # Update workflow step with response
        await WorkflowStepDataManager(self.session).update_by_fields(db_workflow_step, {"data": data})

        # Update progress in workflow
        model_security_scan_response["progress_type"] = BudServeWorkflowStepEventName.MODEL_SECURITY_SCAN_EVENTS.value

        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"progress": model_security_scan_response}
        )

    async def _perform_model_security_scan_request(self, workflow_id: UUID, data: Dict, current_user_id: UUID) -> None:
        """Perform model security scan request."""
        model_security_scan_endpoint = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_model_app_id}/method/model-info/scan"
        )

        # Retrieve model local path
        model_id = data.get("model_id")
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            Model, {"id": model_id, "status": ModelStatusEnum.ACTIVE}
        )
        local_path = db_model.local_path

        model_security_scan_request = {
            "model_path": local_path,
            "notification_metadata": {
                "name": BUD_INTERNAL_WORKFLOW,
                "subscriber_ids": str(current_user_id),
                "workflow_id": str(workflow_id),
            },
            "source_topic": f"{app_settings.source_topic}",
        }
        logger.debug(f"Model security scan payload: {model_security_scan_request}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(model_security_scan_endpoint, json=model_security_scan_request) as response:
                    response_data = await response.json()
                    if response.status >= 400:
                        raise ClientException("unable to perform model security scan request")

                    return response_data
        except ClientException as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to perform model security scan request: {e}")
            raise ClientException("unable to perform model security scan request") from e

    async def create_scan_result_from_notification_event(self, payload: NotificationPayload) -> None:
        """Create a local model security scan result from notification event."""
        logger.debug("Received event for creating local model security scan result")

        # Get workflow steps
        workflow_id = payload.workflow_id
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})
        logger.debug(f"Retrieved workflow: {db_workflow.id}")

        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Define the keys required for endpoint creation
        keys_of_interest = [
            "model_id",
        ]

        # from workflow steps extract necessary information
        required_data = {}
        for db_workflow_step in db_workflow_steps:
            for key in keys_of_interest:
                if key in db_workflow_step.data:
                    required_data[key] = db_workflow_step.data[key]
        logger.debug("Collected required data from workflow steps")

        # Get model
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            Model, {"id": required_data["model_id"], "status": ModelStatusEnum.ACTIVE}
        )
        local_path = db_model.local_path
        logger.debug(f"Local path: {local_path}")

        # Get model scan result
        scan_result = payload.content.result["scan_result"]
        logger.debug(f"Scan result: {scan_result}")

        # Parse necessary data from scan result
        total_issues_by_severity = scan_result.get("total_issues_by_severity", {})
        low_severity_count = total_issues_by_severity.get("LOW", 0)
        medium_severity_count = total_issues_by_severity.get("MEDIUM", 0)
        high_severity_count = total_issues_by_severity.get("HIGH", 0)
        critical_severity_count = total_issues_by_severity.get("CRITICAL", 0)

        model_issues = scan_result.get("model_issues", [])
        grouped_issues = await self._group_model_issues(model_issues, local_path)

        # Determine overall scan status
        overall_scan_status = await self.determine_overall_scan_status(
            low_severity_count, medium_severity_count, high_severity_count, critical_severity_count
        )

        # Create model security scan result
        model_security_scan_result = ModelSecurityScanResultCreate(
            model_id=db_model.id,
            status=overall_scan_status,
            total_issues=scan_result.get("total_issues", 0),
            total_scanned_files=scan_result.get("total_scanned", 0),
            total_skipped_files=scan_result.get("total_skipped_files", 0),
            scanned_files=scan_result.get("scanned_files", []),
            low_severity_count=low_severity_count,
            medium_severity_count=medium_severity_count,
            high_severity_count=high_severity_count,
            critical_severity_count=critical_severity_count,
            model_issues=grouped_issues,
        )
        logger.debug("Parsed model security scan result")

        # Check if model security scan result already exists
        db_model_security_scan_result = await ModelSecurityScanResultDataManager(self.session).retrieve_by_fields(
            ModelSecurityScanResultModel, {"model_id": db_model.id}, missing_ok=True
        )

        if db_model_security_scan_result:
            logger.debug("Model security scan result already exists. Updating it.")
            db_model_security_scan_result = await ModelSecurityScanResultDataManager(self.session).update_by_fields(
                db_model_security_scan_result,
                model_security_scan_result.model_dump(),
            )
        else:
            logger.debug("Model security scan result does not exist. Creating it.")
            db_model_security_scan_result = await ModelSecurityScanResultDataManager(self.session).insert_one(
                ModelSecurityScanResultModel(**model_security_scan_result.model_dump())
            )

        # Update workflow current step
        current_step_number = db_workflow.current_step + 1
        workflow_current_step = current_step_number

        # Update workflow step with model security scan result id
        db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
            db_workflow.id, current_step_number, {"security_scan_result_id": str(db_model_security_scan_result.id)}
        )

        # Update workflow current step
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow,
            {"current_step": workflow_current_step},
        )

        # Mark scan_verified according to overall scan status
        scan_verified = True if overall_scan_status == ModelSecurityScanStatusEnum.SAFE else False

        db_model = await ModelDataManager(self.session).update_by_fields(
            db_model,
            {"scan_verified": scan_verified},
        )

        # Get dummy leaderboard data
        leaderboard_data = await self.get_leaderboard()

        # Update workflow current step
        current_step_number = db_workflow.current_step + 1
        workflow_current_step = current_step_number

        # Create workflow step to store model scan result
        db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
            workflow_id, current_step_number, {"leaderboard": leaderboard_data}
        )

        # Update workflow current step and status
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow,
            {"current_step": workflow_current_step, "status": WorkflowStatusEnum.COMPLETED},
        )

        # Send notification to workflow creator
        model_icon = await ModelServiceUtil(self.session).get_model_icon(db_model)
        notification_request = (
            NotificationBuilder()
            .set_content(
                title=db_model.name,
                message="Scan is Completed",
                icon=model_icon,
                tag=overall_scan_status.value,
                result=NotificationResult(target_id=db_model.id, target_type="model").model_dump(
                    exclude_none=True, exclude_unset=True
                ),
            )
            .set_payload(workflow_id=str(db_workflow.id), type=NotificationTypeEnum.MODEL_SCAN_SUCCESS.value)
            .set_notification_request(subscriber_ids=[str(db_workflow.created_by)])
            .build()
        )
        await BudNotifyService().send_notification(notification_request)

    async def _group_model_issues(self, model_issues: list, local_path: str) -> list[ModelIssue]:
        """Group model issues by severity."""
        grouped_issues = {}

        for model_issue in model_issues:
            severity = model_issue["severity"].lower()

            # NOTE: changed this formatting to budmodel
            # Clean up the source path by removing local_path prefix
            # source = model_issue["source"]
            # if source.startswith(local_path):
            #     source = source[len(local_path) :].lstrip("/")

            # Group issues by severity
            if severity not in grouped_issues:
                grouped_issues[severity] = []
            grouped_issues[severity].append(
                ModelIssue(
                    title=model_issue["title"],
                    severity=severity,
                    description=model_issue["description"],
                    source=model_issue["source"],
                ).model_dump()
            )

        return grouped_issues

    async def get_leaderboard(self) -> dict:
        """Get leaderboard for a model."""
        return {
            "headers": {
                "evaluation_type": "Evaluation type",
                "dataset": "Data Set",
                "model_1": "Selected Model",
                "model_2": "Model 2",
            },
            "rows": [
                {"evaluation_type": "IFEval", "dataset": "Reasoning", "model_1": 65.1, "model_2": 65.1},
                {"evaluation_type": "BBH", "dataset": "MMLU", "model_1": 46.9, "model_2": 46.9},
                {"evaluation_type": "Model", "dataset": "Factuality", "model_1": 78.4, "model_2": 78.4},
                {"evaluation_type": "Tags", "dataset": "Reasoning", "model_1": 60.8, "model_2": 60.8},
            ],
        }

    @staticmethod
    async def determine_overall_scan_status(
        low_count: int, medium_count: int, high_count: int, critical_count: int
    ) -> ModelSecurityScanStatusEnum:
        """Determine the overall security scan status based on issue counts.

        Args:
            low_count: Number of low severity issues
            medium_count: Number of medium severity issues
            high_count: Number of high severity issues
            critical_count: Number of critical severity issues

        Returns:
            ModelSecurityScanStatusEnum: The overall status based on the highest severity with issues
        """
        if critical_count > 0:
            return ModelSecurityScanStatusEnum.CRITICAL
        elif high_count > 0:
            return ModelSecurityScanStatusEnum.HIGH
        elif medium_count > 0:
            return ModelSecurityScanStatusEnum.MEDIUM
        elif low_count > 0:
            return ModelSecurityScanStatusEnum.LOW
        else:
            return ModelSecurityScanStatusEnum.SAFE  # Default to SAFE if no issues are found

    @staticmethod
    def save_author_logo(img_url: str) -> str:
        """Downloads and saves the logo from the given image URL locally with a unique name.

        Args:
            img_url (str): The URL of the logo image.

        Returns:
            str: The local file path of the saved logo.
        """
        # Create logo directory if it doesn't exist
        logo_dir = os.path.join(app_settings.icon_dir, HF_AUTHORS_DIR)
        os.makedirs(logo_dir, exist_ok=True)

        try:
            # Parse the URL
            parsed_url = urlparse(img_url)
            path_parts = parsed_url.path.strip("/").split("/")

            # Use the last two parts as the filename
            logo_name = f"{path_parts[-2]}_{path_parts[-1]}"

        except Exception:
            # Fallback to entire URL path with `/` replaced by `_`
            logo_name = parsed_url.path.strip("/").replace("/", "_")

        # Ensure the filename ends with .png
        if not logo_name.endswith(".png"):
            logo_name += ".png"

        # Construct the local logo path
        logo_path = os.path.join(logo_dir, logo_name)
        icon_index = logo_path.find("/icons/")
        formatted_logo_path = logo_path[icon_index + 1 :]

        # Check if the logo already exists
        if os.path.exists(logo_path):
            logger.debug(f"Logo already saved at: {logo_path}")
            return formatted_logo_path

        # Download and save the image
        try:
            response = requests.get(img_url)
            response.raise_for_status()

            # Save the image locally
            with open(logo_path, "wb") as f:
                f.write(response.content)

            logger.debug(f"Logo saved at: {logo_path}")
            return formatted_logo_path

        except Exception as e:
            logger.debug(f"Failed to download logo: {e}")
            return ""


class CloudModelService(SessionMixin):
    """Cloud model service."""

    async def get_all_cloud_models(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[CloudModel], int]:
        """Get all cloud models."""
        db_cloud_models, count = await CloudModelDataManager(self.session).get_all_cloud_models(
            offset, limit, filters, order_by, search
        )

        # convert db_cloud_models to cloud model list response
        db_cloud_models_response = []
        for db_cloud_model in db_cloud_models:
            model_response = ModelResponse.model_validate(db_cloud_model)
            db_cloud_models_response.append(ModelListResponse(model=model_response))

        return db_cloud_models_response, count

    async def get_all_recommended_tags(
        self,
        offset: int = 0,
        limit: int = 10,
    ) -> Tuple[List[CloudModel], int]:
        """Get all cloud models."""
        return await CloudModelDataManager(self.session).get_all_recommended_tags(offset, limit)


class ModelService(SessionMixin):
    """Model service."""

    async def retrieve_model(self, model_id: UUID) -> ModelDetailSuccessResponse:
        """Retrieve model details by model ID."""
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            Model, {"id": model_id, "status": ModelStatusEnum.ACTIVE}
        )

        # Get base model relation count
        model_tree_count = await ModelDataManager(self.session).get_model_tree_count(db_model.uri)
        base_model_relation_count = {row.base_model_relation.value: row.count for row in model_tree_count}
        model_tree = ModelTree(
            adapters_count=base_model_relation_count.get(BaseModelRelationEnum.ADAPTER.value, 0),
            finetunes_count=base_model_relation_count.get(BaseModelRelationEnum.FINETUNE.value, 0),
            merges_count=base_model_relation_count.get(BaseModelRelationEnum.MERGE.value, 0),
            quantizations_count=base_model_relation_count.get(BaseModelRelationEnum.QUANTIZED.value, 0),
        )

        db_endpoint_count = await ModelDataManager(self.session).get_count_by_fields(
            EndpointModel, fields={"model_id": model_id}, exclude_fields={"status": EndpointStatusEnum.DELETED}
        )

        return ModelDetailSuccessResponse(
            model=db_model,
            model_tree=model_tree,
            scan_result=db_model.model_security_scan_result,
            endpoints_count=db_endpoint_count,
            message="model retrieved successfully",
            code=status.HTTP_200_OK,
            object="model.get",
        ).to_http_response()

    async def list_model_tags(self, name: str, offset: int = 0, limit: int = 10) -> tuple[list[Tag], int]:
        """Search model tags by name with pagination."""
        tags_result, count = await ModelDataManager(self.session).list_model_tags(name, offset, limit)
        tags = [Tag(name=row.name, color=row.color) for row in tags_result]

        return tags, count

    async def list_model_tasks(self, name: str, offset: int = 0, limit: int = 10) -> tuple[list[Task], int]:
        """Search model tasks by name with pagination."""
        tasks_result, count = await ModelDataManager(self.session).list_model_tasks(name, offset, limit)
        tasks = [Task(name=row.name, color=row.color) for row in tasks_result]

        return tasks, count

    async def get_all_active_models(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[Model], int]:
        """Get all active models."""
        filters_dict = filters

        results, count = await ModelDataManager(self.session).get_all_models(
            offset, limit, filters_dict, order_by, search
        )

        # Parse the results to model list response
        db_models_response = []
        for result in results:
            model_response = ModelResponse.model_validate(result[0])
            db_models_response.append(ModelListResponse(model=model_response, endpoints_count=result[1]))

        return db_models_response, count

    async def list_all_model_authors(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[str], int]:
        """Search author by name with pagination support."""
        filters["status"] = ModelStatusEnum.ACTIVE
        db_models, count = await ModelDataManager(self.session).list_all_model_authors(
            offset, limit, filters, order_by, search
        )
        db_authors = [model.author for model in db_models]

        return db_authors, count

    async def edit_model(self, model_id: UUID, data: Dict[str, Any], current_user_id: UUID) -> None:
        """Edit cloud model by validating and updating specific fields, and saving an uploaded file if provided."""
        logger.debug(f"edit recieved data: {data}")
        # Retrieve existing model
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            model=Model, fields={"id": model_id, "status": ModelStatusEnum.ACTIVE}
        )

        if data.get("name"):
            duplicate_model = await ModelDataManager(self.session).retrieve_by_fields(
                model=Model,
                fields={"name": data["name"], "status": ModelStatusEnum.ACTIVE},
                exclude_fields={"id": model_id},
                missing_ok=True,
                case_sensitive=False,
            )
            if duplicate_model:
                raise ClientException("Model name already exists")

        if data.get("icon") and db_model.provider_type in [
            ModelProviderTypeEnum.CLOUD_MODEL,
            ModelProviderTypeEnum.HUGGING_FACE,
        ]:
            data.pop("icon")

        # Handle file upload if provided
        # TODO: consider dapr local storage
        if data.get("license_file"):
            # If a file is provided, save it locally and update the DB with the local path
            file = data.pop("license_file")

            # Save to minio
            license_object_name = await self._save_license_to_minio(file, model_id)

            # Create or update license entry
            await self._create_or_update_license_entry(
                model_id, file.filename, license_object_name, current_user_id, ModelLicenseObjectTypeEnum.MINIO
            )

        elif data.get("license_url"):
            # If a license URL is provided, store the URL in the DB instead of the file path
            license_url = data.pop("license_url")
            filename = license_url.split("/")[-1]  # Extract filename from the URL
            filename = filename if filename else "sample license"
            await self._create_or_update_license_entry(
                model_id, filename, str(license_url), current_user_id, ModelLicenseObjectTypeEnum.URL
            )  # TODO: modify filename arg when license service implemented

        # Add papers if provided
        if isinstance(data.get("paper_urls"), list):
            paper_urls = data.pop("paper_urls")
            await self._update_papers(model_id, paper_urls)

        # Update model with validated data
        await ModelDataManager(self.session).update_by_fields(db_model, data)

    async def _save_license_to_minio(self, file: UploadFile, model_id: UUID) -> str:
        """Save uploaded file and return file path."""
        # Check if the license file already exists in minio
        minio_store = ModelStore()
        license_object_prefix = f"{model_id}"
        try:
            # is_minio_object_exists = minio_store.check_file_exists(app_settings.minio_model_bucket, license_object_prefix)
            minio_store.remove_objects(app_settings.minio_model_bucket, license_object_prefix, recursive=True)

            # Download file to a temp folder and save it to minio
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                license_object_name = f"{model_id}/{file.filename}"

                # Upload file to minio
                minio_store.upload_file(app_settings.minio_model_bucket, file_path, license_object_name)

            return license_object_name
        except MinioException as e:
            logger.exception(f"Error uploading file to minio: {e}")
            raise ClientException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Unable to save license file"
            ) from e

    async def _create_or_update_license_entry(
        self,
        model_id: UUID,
        filename: str,
        license_url: str,
        current_user_id: UUID,
        data_type: ModelLicenseObjectTypeEnum,
    ) -> None:
        """Create or update a license entry in the database."""
        # Set license source
        if data_type == ModelLicenseObjectTypeEnum.MINIO:
            minio_store = ModelStore()
            license_source = minio_store.get_object_url(app_settings.minio_model_bucket, license_url)
        else:
            license_source = license_url

        # Check if a license entry with the given model_id exists
        existing_license = await ModelLicensesDataManager(self.session).retrieve_by_fields(
            ModelLicenses, fields={"model_id": model_id}, missing_ok=True
        )

        if existing_license:
            logger.debug(f"existing license: {existing_license}")

            update_license_data = {
                "name": filename,
                "url": license_url,
                "data_type": data_type,
                "faqs": None,
                "description": None,
                "suitability": None,
                "license_type": None,
            }

            # Update database
            await ModelLicensesDataManager(self.session).update_by_fields(existing_license, update_license_data)

            # Execute license faqs workflow
            await self.fetch_license_faqs(model_id, existing_license.id, current_user_id, license_source)
        else:
            # Create a new license entry
            license_entry = ModelLicensesModel(
                id=uuid4(),
                name=filename,
                url=license_url,
                data_type=data_type,
                model_id=model_id,
                faqs=None,
                description=None,
                suitability=None,
                license_type=None,
            )

            # Update database
            await ModelLicensesDataManager(self.session).insert_one(
                ModelLicenses(**license_entry.model_dump(exclude_unset=True))
            )

            # Execute license faqs workflow
            await self.fetch_license_faqs(model_id, license_entry.id, current_user_id, license_source)

    async def _update_papers(self, model_id: UUID, paper_urls: list[str]) -> None:
        """Update paper entries for the given model by adding new URLs and removing old ones."""
        # Fetch existing paper URLs for the model
        existing_papers = await ModelDataManager(self.session).get_all_by_fields(
            model=PaperPublished, fields={"model_id": model_id}
        )
        existing_urls = {paper.url for paper in existing_papers}

        # Determine URLs to add and remove
        input_urls = set(paper_urls)
        urls_to_add = input_urls - existing_urls
        urls_to_remove = existing_urls - input_urls
        logger.debug(
            f"paper info: {input_urls}, existing urls: {existing_urls}, urls_to_add: {urls_to_add}, urls_to_remove: {urls_to_remove}"
        )

        # Add new paper URLs
        if urls_to_add:
            urls_to_add = [
                PaperPublished(id=uuid4(), title="Untitled Research Paper", url=str(paper_url), model_id=model_id)
                for paper_url in urls_to_add
            ]
            await PaperPublishedDataManager(self.session).insert_all(urls_to_add)

        # Remove old paper URLs
        if urls_to_remove:
            # Delete all matching entries for given URLs and model_id in one query
            await PaperPublishedDataManager(self.session).delete_paper_by_urls(
                model_id=model_id, paper_urls={"url": urls_to_remove}
            )

    async def cancel_model_deployment_workflow(self, workflow_id: UUID) -> None:
        """Cancel model deployment workflow."""
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": db_workflow.id}
        )

        # Define the keys required for endpoint creation
        keys_of_interest = [
            BudServeWorkflowStepEventName.BUDSERVE_CLUSTER_EVENTS.value,
        ]

        # from workflow steps extract necessary information
        required_data = {}
        for db_workflow_step in db_workflow_steps:
            for key in keys_of_interest:
                if key in db_workflow_step.data:
                    required_data[key] = db_workflow_step.data[key]
        logger.debug("Collected required data from workflow steps")

        if required_data.get(BudServeWorkflowStepEventName.BUDSERVE_CLUSTER_EVENTS.value) is None:
            raise ClientException("Model deployment process has not been initiated")

        budserve_cluster_response = required_data.get(BudServeWorkflowStepEventName.BUDSERVE_CLUSTER_EVENTS.value)
        dapr_workflow_id = budserve_cluster_response.get("workflow_id")

        try:
            await self._perform_cancel_model_deployment_request(dapr_workflow_id)
        except ClientException as e:
            raise e

    async def _perform_cancel_model_deployment_request(self, workflow_id: str) -> Dict:
        """Perform cancel model deployment request to bud_cluster app.

        Args:
            workflow_id: The ID of the workflow to cancel.
        """
        cancel_model_deployment_endpoint = f"{app_settings.dapr_base_url}v1.0/invoke/{app_settings.bud_cluster_app_id}/method/deployment/cancel/{workflow_id}"

        logger.debug(f"Performing cancel model deployment request to budcluster {cancel_model_deployment_endpoint}")
        try:
            async with aiohttp.ClientSession() as session, session.post(cancel_model_deployment_endpoint) as response:
                response_data = await response.json()
                if response.status != 200 or response_data.get("object") == "error":
                    logger.error(f"Failed to cancel model deployment: {response.status} {response_data}")
                    raise ClientException(
                        "Failed to cancel model deployment", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )

                logger.debug("Successfully cancelled model deployment")
                return response_data
        except Exception as e:
            logger.exception(f"Failed to send cancel model deployment request: {e}")
            raise ClientException(
                "Failed to cancel model deployment", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from e

    async def delete_active_model(self, model_id: UUID) -> Model:
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            Model, fields={"id": model_id, "status": ModelStatusEnum.ACTIVE}
        )

        # Check for active endpoint
        db_endpoints = await EndpointDataManager(self.session).retrieve_by_fields(
            EndpointModel,
            fields={"model_id": model_id},
            exclude_fields={"status": EndpointStatusEnum.DELETED},
            missing_ok=True,
        )

        # Raise error if model has active endpoint
        if db_endpoints:
            raise ClientException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete model because it has active endpoint.",
            )

        if db_model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
            db_cloud_model = await CloudModelDataManager(self.session).retrieve_by_fields(
                CloudModel,
                fields={
                    "status": CloudModelStatusEnum.ACTIVE,
                    "source": db_model.source,
                    "uri": db_model.uri,
                    "provider_id": db_model.provider_id,
                    "is_present_in_model": True,
                },
                missing_ok=True,
            )

            # If cloud model is added from seeded model, update is_present_in_model to False
            if db_cloud_model:
                db_cloud_model = await CloudModelDataManager(self.session).update_by_fields(
                    db_cloud_model, fields={"is_present_in_model": False}
                )
            else:
                logger.warning("This cloud model is not added from seeded cloud models")

        else:
            await self._perform_model_deletion_request(db_model.local_path)
            logger.debug(f"Model deletion successful for {db_model.local_path}")

        # Delete existing license from minio
        if db_model.model_licenses:
            logger.debug(f"Deleting license from minio for model {db_model.id}")
            if db_model.model_licenses.data_type == ModelLicenseObjectTypeEnum.MINIO:
                minio_store = ModelStore()
                minio_store.remove_objects(app_settings.minio_model_bucket, f"{db_model.id}", recursive=True)

            # Delete license from db
            await ModelLicensesDataManager(self.session).delete_by_fields(ModelLicenses, {"model_id": db_model.id})

        db_model = await ModelDataManager(self.session).update_by_fields(db_model, {"status": ModelStatusEnum.DELETED})

        # Remove from recommended models
        await ModelClusterRecommendedDataManager(self.session).delete_by_fields(
            ModelClusterRecommendedModel, {"model_id": db_model.id}
        )
        logger.debug(f"Model recommended cluster data for model {db_model.id} deleted")

        return db_model

    async def _perform_model_deletion_request(self, local_path: str) -> None:
        """Perform model deletion request."""
        model_deletion_endpoint = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_model_app_id}/method/model-info/local-models"
        )

        params = {"path": local_path}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(model_deletion_endpoint, params=params) as response:
                    if response.status >= 400:
                        raise ClientException("Unable to perform model deletion")

        except ClientException as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to perform model deletion request: {e}")
            raise ClientException("Unable to perform local model deletion") from e

    async def fetch_license_faqs(
        self, model_id: UUID, license_id: UUID, current_user_id: UUID, license_source: str
    ) -> WorkflowModel:
        """Fetch license faqs of a license by path or url.

        Args:
            model_id: The ID of the model corresponding to license source.
            license_source: file path or web url of license file.
        """
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            Model, fields={"id": model_id}, exclude_fields={"status": ModelStatusEnum.DELETED}
        )

        current_step_number = 1

        # Retrieve or create workflow
        workflow_create = WorkflowUtilCreate(
            workflow_type=WorkflowTypeEnum.LICENSE_FAQ_FETCH,
            title="Model License FAQS",
            total_steps=current_step_number,
            icon=db_model.icon,
            tag="Model License FAQS",
            visibility=VisibilityEnum.INTERNAL,
        )
        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id=None, workflow_data=workflow_create, current_user_id=current_user_id
        )
        logger.debug(f"model license faq {db_workflow.id} created")

        # Perform license faq fetch request to bud_model app
        try:
            bud_model_response = await self._perform_license_faq_fetch_request(
                license_source, current_user_id, db_workflow.id
            )
        except ClientException as e:
            await WorkflowDataManager(self.session).update_by_fields(
                db_workflow, {"status": WorkflowStatusEnum.FAILED}
            )
            raise e

        # Add payload dict to response
        for step in bud_model_response["steps"]:
            step["payload"] = {}

        license_faq_events = {
            BudServeWorkflowStepEventName.LICENSE_FAQ_EVENTS.value: bud_model_response,
            "license_id": str(license_id),
            "model_id": str(model_id),
        }

        # Insert step details in db
        db_workflow_step = await WorkflowStepDataManager(self.session).insert_one(
            WorkflowStepModel(
                workflow_id=db_workflow.id,
                step_number=current_step_number,
                data=license_faq_events,
            )
        )
        logger.debug(f"Created workflow step {current_step_number} for workflow {db_workflow.id}")

        # Update progress in workflow
        bud_model_response["progress_type"] = BudServeWorkflowStepEventName.LICENSE_FAQ_EVENTS.value
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"progress": bud_model_response, "current_step": current_step_number}
        )

        return db_workflow

    async def _perform_license_faq_fetch_request(
        self, license_source: str, current_user_id: UUID, workflow_id: UUID
    ) -> Dict:
        """Perform license faqs fetch request to bud_model app.

        Args:
            license_source: The source of license, can be file path or url.
        """
        license_faq_fetch_endpoint = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_model_app_id}/method/model-info/license-faq"
        )

        payload = {
            "license_source": str(license_source),
            "notification_metadata": {
                "name": BUD_INTERNAL_WORKFLOW,
                "subscriber_ids": str(current_user_id),
                "workflow_id": str(workflow_id),
            },
            "source_topic": f"{app_settings.source_topic}",
        }

        logger.debug(f"Performing license faqs fetch request to budmodel {payload}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(license_faq_fetch_endpoint, json=payload) as response:
                    response_data = await response.json()
                    if response.status != 200:
                        logger.error(f"Failed to fetch license faqs: {response.status} {response_data}")
                        raise ClientException(
                            "Failed to fetch license faqs", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                        )

                    logger.debug(f"Successfully fetched license faqs from budmodel{response_data}")
                    return response_data
        except Exception as e:
            logger.exception(f"Failed to send license faqs fetch request: {e}")
            raise ClientException(
                "Failed to fetch license faqs", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from e

    async def update_license_faqs_from_notification_event(self, payload: NotificationPayload) -> None:
        """Update license faqs from notification event."""
        logger.debug("Received event for fetching license faqs")

        # Get workflow and steps
        workflow_id = payload.workflow_id
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Define the keys required for updating faqs
        keys_of_interest = ["model_id", "license_id"]

        # from workflow steps extract necessary information
        required_data = {}
        for db_workflow_step in db_workflow_steps:
            for key in keys_of_interest:
                if key in db_workflow_step.data:
                    required_data[key] = db_workflow_step.data[key]

        logger.debug("Collected required data from workflow steps")

        # Retrieve cluster from db
        db_license = await ModelLicensesDataManager(self.session).retrieve_by_fields(
            ModelLicenses,
            fields={"id": required_data["license_id"], "model_id": required_data["model_id"]},
            missing_ok=True,
        )
        logger.debug(f"license retrieved successfully: {db_license.id}")

        # update license details
        license_details = payload.content.result["license_details"]

        license_faqs = normalize_value(license_details.get("faqs", []))
        updated_license_faqs = []
        if license_faqs:
            for faq in license_faqs:
                faq_description = " ".join(faq.get("reason", [])).strip()
                impact = faq.get("impact", "")
                answer = "YES" if impact == "POSITIVE" else "NO"
                updated_license_faqs.append(
                    {
                        "question": faq.get("question"),
                        "description": faq_description,
                        "answer": answer,
                    }
                )

        license_data = {
            "name": normalize_value(license_details.get("name", "license")),
            "license_type": normalize_value(license_details.get("type")),
            "description": normalize_value(license_details.get("type_description")),
            "suitability": normalize_value(license_details.get("type_suitability")),
            "faqs": updated_license_faqs,
        }

        db_license = await ModelLicensesDataManager(self.session).update_by_fields(db_license, license_data)
        logger.debug(f"updated FAQs for license {db_license.id}")

        # Mark workflow as completed
        await WorkflowDataManager(self.session).update_by_fields(db_workflow, {"status": WorkflowStatusEnum.COMPLETED})
        logger.debug(f"Workflow {db_workflow.id} marked as completed")

    async def list_leaderboards(
        self, model_id: UUID, table_source: Literal["cloud_model", "model"], k: int
    ) -> List[LeaderboardTable]:
        """List leaderboards of specific model by uri.

        Args:
            model_id: The ID of the model.
            table_source: The source of the model.
            k: The maximum number of leaderboards to return.

        Returns:
            The leaderboards of the model.
        """
        if table_source == "cloud_model":
            db_model = await CloudModelDataManager(self.session).retrieve_by_fields(
                CloudModel, fields={"id": model_id, "status": CloudModelStatusEnum.ACTIVE}
            )
        elif table_source == "model":
            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model, fields={"id": model_id, "status": ModelStatusEnum.ACTIVE}
            )
        else:
            raise ClientException("Invalid table source", status_code=status.HTTP_400_BAD_REQUEST)

        model_uri = db_model.uri

        # Fetch leaderboards from bud_model app
        bud_model_response = await self._perform_bud_model_leaderboard_fetch_request(model_uri, k)

        return await self._parse_leaderboard_response(bud_model_response, model_uri)

    async def _parse_leaderboard_response(
        self, bud_model_response: Dict, selected_model_uri: str
    ) -> List[LeaderboardTable]:
        """Parse leaderboard response from bud_model app."""
        # Validate response
        bud_model_leaderboards = bud_model_response.get("leaderboards", [])
        if not bud_model_leaderboards:
            logger.debug("No leaderboard found for model %s", selected_model_uri)
            return []

        # Find the leaderboard for the selected model
        selected_model_leaderboard = None
        for leaderboard in bud_model_leaderboards:
            if leaderboard.get("model_info", {}).get("uri") == selected_model_uri:
                selected_model_leaderboard = leaderboard
                break
        if not selected_model_leaderboard:
            logger.debug("No leaderboard found for selected model %s", selected_model_uri)
            return []

        # Ignore fields with None values
        valid_fields = []
        for benchmark in selected_model_leaderboard.get("benchmarks", []):
            valid_fields.append(benchmark.get("eval_name"))
        # valid_fields = [key for key in selected_model_leaderboard.keys() if key != "model_info"]
        logger.debug("Valid fields: %s", valid_fields)

        leaderboard_tables: List[LeaderboardTable] = []

        for leaderboard in bud_model_leaderboards:
            model_info = leaderboard.get("model_info", {})
            bud_model_benchmarks = leaderboard.get("benchmarks", [])
            # Create model info
            model = LeaderboardModelInfo(
                uri=model_info.get("uri"),
                model_size=model_info.get("num_params"),
                is_selected=model_info.get("uri") == selected_model_uri,
            )

            benchmarks = {}
            for bud_model_benchmark in bud_model_benchmarks:
                field = bud_model_benchmark.get("eval_name")
                if field in valid_fields:
                    label_alternative = bud_model_benchmark.get("eval_label")
                    benchmarks[field] = LeaderboardBenchmark(
                        type=BENCHMARK_FIELDS_TYPE_MAPPER.get(field, None),
                        value=bud_model_benchmark.get("eval_score"),
                        label=BENCHMARK_FIELDS_LABEL_MAPPER.get(field, label_alternative),
                    )
            if benchmarks:
                leaderboard_tables.append(LeaderboardTable(model=model, benchmarks=benchmarks))

        return leaderboard_tables

    async def _perform_bud_model_leaderboard_fetch_request(self, uri: str, k: int) -> Dict:
        """Perform leaderboard fetch request to bud_model app.

        Args:
            uri: The uri of the model.
            k: The maximum number of leaderboards to return.
        """
        leaderboard_fetch_url = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_model_app_id}/method/leaderboard/model-params"
        )

        query_params = {"model_uri": uri, "k": k}

        logger.debug(f"Performing leaderboard fetch request to budmodel {query_params}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(leaderboard_fetch_url, params=query_params) as response:
                    response_data = await response.json()
                    if response.status != 200:
                        logger.error(f"Failed to fetch leaderboards: {response.status} {response_data}")
                        raise ClientException("Failed to fetch leaderboards")

                    logger.debug("Successfully fetched leaderboards from budmodel")
                    return response_data
        except Exception as e:
            logger.exception(f"Failed to send leaderboard fetch request: {e}")
            raise ClientException("Failed to fetch leaderboards") from e

    async def get_top_leaderboards(self, benchmarks: List[str], limit: int) -> TopLeaderboard:
        """Get top leaderboards of a model by uri.

        Args:
            benchmarks: The benchmarks to return.
            limit: The maximum number of leaderboards to return.
        """
        # Get all active model uris
        db_models = await ModelDataManager(self.session).get_all_by_fields(Model, {"status": ModelStatusEnum.ACTIVE})
        db_model_uris = [model.uri for model in db_models]

        # Fetch top leaderboards from bud_model app
        bud_model_response = await self._perform_top_leaderboard_by_uri_request(db_model_uris, benchmarks, limit)

        bud_model_leaderboards = bud_model_response.get("leaderboards", [])

        if len(bud_model_leaderboards) == 0:
            return []

        # Get model info by uris
        leaderboard_uris = [leaderboard.get("uri") for leaderboard in bud_model_leaderboards]
        db_models = await ModelDataManager(self.session).get_models_by_uris(leaderboard_uris)

        # If no models found, return empty list
        if len(db_models) == 0:
            return []

        db_model_info = await self._get_model_info_by_uris(db_models)

        # Parse top leaderboard response
        return await self._parse_top_leaderboard_response(bud_model_leaderboards, db_model_info, benchmarks)

    async def _parse_top_leaderboard_response(
        self, bud_model_leaderboards: List[Dict], db_model_info: Dict, fields: List[str]
    ) -> List[TopLeaderboard]:
        """Parse top leaderboard response from bud_model app."""
        result = []

        # Iterate over leaderboards
        for leaderboard in bud_model_leaderboards:
            leaderboard_model_uri = leaderboard.get("uri")

            # If model not found, skip
            if leaderboard_model_uri not in db_model_info:
                continue

            benchmarks = []
            for bud_model_benchmark in leaderboard.get("benchmarks", []):
                field = bud_model_benchmark.get("eval_name")
                if field in fields:
                    label_alternative = bud_model_benchmark.get("eval_label")
                    benchmarks.append(
                        TopLeaderboardBenchmark(
                            field=field,
                            value=bud_model_benchmark.get("eval_score"),
                            type=BENCHMARK_FIELDS_TYPE_MAPPER.get(field, None),
                            label=BENCHMARK_FIELDS_LABEL_MAPPER.get(field, label_alternative),
                        ).model_dump()
                    )
            result.append(
                TopLeaderboard(
                    benchmarks=benchmarks,
                    name=db_model_info.get(leaderboard_model_uri, {}).get("name"),
                    provider_type=db_model_info.get(leaderboard_model_uri, {}).get("provider_type"),
                )
            )

        return result

    async def _get_model_info_by_uris(self, db_models: List[Model]) -> Dict:
        """Get model info by uris."""
        model_info = {}

        # Collect name and provider type for each model
        for db_model in db_models:
            model_info[db_model.uri] = {
                "name": db_model.name,
                "provider_type": db_model.provider_type,
            }

        return model_info

    async def _perform_top_leaderboard_by_uri_request(
        self, uris: List[str], benchmark_fields: List[str], k: int
    ) -> Dict:
        """Perform top leaderboard fetch request to bud_model app.

        Args:
            uris: The uris of the models.
            benchmark_fields: The benchmarks to return.
            k: The maximum number of leaderboards to return.
        """
        bud_model_endpoint = f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_model_app_id}/method/leaderboard/models/compare"

        query_params = {"model_uris": uris, "benchmark_fields": benchmark_fields, "k": k}

        logger.debug(f"Performing top leaderboard by uris request to budmodel {query_params}")
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(bud_model_endpoint, params=query_params) as response,
            ):
                response_data = await response.json()
                if response.status != 200:
                    logger.error(f"Failed to fetch top leaderboards: {response.status} {response_data}")
                    raise ClientException("Failed to fetch top leaderboards")

                logger.debug("Successfully fetched top leaderboards from budmodel")
                return response_data
        except Exception as e:
            logger.exception(f"Failed to send top leaderboard by uris request: {e}")
            raise ClientException("Failed to fetch top leaderboards") from e

    async def get_leaderboard_by_model_uris(self, model_uris: List[str]) -> dict:
        """Service method to fetch and parse leaderboard for a given model URI.

        Args:
            model_uris (str): Model URIs to query leaderboard for.

        Returns:
            List[LeaderboardTable]: Parsed leaderboard table data.
        """
        # Fetch leaderboard data from bud_model app
        bud_model_response = await self._perform_leaderboard_by_uris_request(model_uris)
        bud_model_leaderboards = bud_model_response.get("leaderboards", {})
        parsed_leaderboards = {}
        for leaderboard in bud_model_leaderboards:
            leaderboard_model_uri = leaderboard.get("uri")

            # If model not found, skip
            if leaderboard_model_uri not in model_uris:
                continue

            benchmarks = []
            for bud_model_benchmark in leaderboard.get("benchmarks", []):
                field = bud_model_benchmark.get("eval_name")
                label_alternative = bud_model_benchmark.get("eval_label")
                benchmarks.append(
                    TopLeaderboardBenchmark(
                        field=field,
                        value=bud_model_benchmark.get("eval_score"),
                        type=BENCHMARK_FIELDS_TYPE_MAPPER.get(field, None),
                        label=BENCHMARK_FIELDS_LABEL_MAPPER.get(field, label_alternative),
                    ).model_dump()
                )
            parsed_leaderboards[leaderboard_model_uri] = benchmarks

        return parsed_leaderboards

    async def _perform_leaderboard_by_uris_request(self, uris: List[str]) -> Dict:
        """Perform top leaderboard fetch request to bud_model app.

        Args:
            uris: The uris of the models.
            benchmark_fields: The benchmarks to return.
            k: The maximum number of leaderboards to return.
        """
        bud_model_endpoint = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_model_app_id}/method/leaderboard/models-uris"
        )

        query_params = {"model_uris": uris}

        logger.debug(f"Performing leaderboard by uri request to budmodel {query_params}")
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(bud_model_endpoint, params=query_params) as response,
            ):
                response_data = await response.json()
                if response.status != 200:
                    logger.error(f"Failed to fetch leaderboards: {response.status} {response_data}")
                    raise ClientException("Failed to fetch leaderboards")

                logger.debug("Successfully fetched leaderboards from budmodel")
                return response_data
        except Exception as e:
            logger.exception(f"Failed to send leaderboard by uris request: {e}")
            raise ClientException("Failed to fetch leaderboards") from e

    async def deploy_model_by_step(
        self,
        current_user_id: UUID,
        step_number: int,
        workflow_id: Optional[UUID] = None,
        workflow_total_steps: Optional[int] = None,
        model_id: Optional[UUID] = None,
        project_id: Optional[UUID] = None,
        cluster_id: Optional[UUID] = None,
        endpoint_name: Optional[str] = None,
        deploy_config: Optional[DeploymentTemplateCreate] = None,
        template_id: Optional[UUID] = None,
        trigger_workflow: bool = False,
        credential_id: Optional[UUID] = None,
        scaling_specification: Optional[ScalingSpecification] = None,
    ) -> EndpointModel:
        """Create workflow steps and execute deployment workflow.

        Arguments availability handled in router pydantic validation
        - If workflow_id is provided, then step_number and either of (model_id, project_id, cluster_id, endpoint_name, replicas) should be provided
        - If workflow_id is not provided, the function will create a new workflow and a new workflow step.
            - Also for workflow step creation needed step_number and either of (model_id, project_id, cluster_id, endpoint_name, replicas)
        """
        current_step_number = step_number

        # Retrieve or create workflow
        workflow_create = WorkflowUtilCreate(
            workflow_type=WorkflowTypeEnum.MODEL_DEPLOYMENT,
            title="Deploying Model",
            total_steps=workflow_total_steps,
            icon=APP_ICONS["general"]["deployment_mono"],
            created_by=current_user_id,
        )

        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id, workflow_create, current_user_id
        )

        # Validate project
        if project_id:
            db_project = await ProjectDataManager(self.session).retrieve_by_fields(
                ProjectModel, {"id": project_id, "status": ProjectStatusEnum.ACTIVE}
            )

            # Update workflow tag
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow, {"tag": db_project.name}
            )

        # Validate model
        if model_id:
            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model, {"id": model_id, "status": ModelStatusEnum.ACTIVE}
            )

            # Update workflow icon
            if db_model.provider_type in [ModelProviderTypeEnum.HUGGING_FACE, ModelProviderTypeEnum.CLOUD_MODEL]:
                db_provider = await ProviderDataManager(self.session).retrieve_by_fields(
                    ProviderModel, {"id": db_model.provider_id}
                )
                model_icon = db_provider.icon
            else:
                model_icon = db_model.icon

            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow, {"icon": model_icon, "title": db_model.name}
            )

        # Validate template
        if template_id:
            db_template = await ModelTemplateDataManager(self.session).retrieve_by_fields(
                ModelTemplateModel, {"id": template_id}
            )

        # Validate cluster
        if cluster_id:
            db_cluster = await ClusterDataManager(self.session).retrieve_by_fields(
                ClusterModel, {"cluster_id": cluster_id}, exclude_fields={"status": ClusterStatusEnum.DELETED}
            )

            if db_cluster.status != ClusterStatusEnum.AVAILABLE:
                logger.error(f"Cluster {cluster_id} is currently not available.")
                raise ClientException("Cluster is not available")

        # Validate credential
        if credential_id:
            db_credential = await ProprietaryCredentialDataManager(self.session).retrieve_by_fields(
                ProprietaryCredentialModel, {"id": credential_id}
            )

        # Update workflow title
        if endpoint_name:
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow, {"title": endpoint_name}
            )

        # Prepare workflow step data
        workflow_step_data = DeploymentWorkflowStepData(
            model_id=model_id,
            project_id=project_id,
            cluster_id=cluster_id,
            endpoint_name=endpoint_name,
            deploy_config=deploy_config,
            template_id=template_id,
            credential_id=credential_id,
            scaling_specification=scaling_specification,
        ).model_dump(exclude_none=True, exclude_unset=True, mode="json")

        # Get workflow steps
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": db_workflow.id}
        )

        # For avoiding another db call for record retrieval, storing db object while iterating over db_workflow_steps
        db_current_workflow_step = None

        if db_workflow_steps:
            db_step_project_id = None
            db_step_endpoint_name = None

            for db_step in db_workflow_steps:
                # Get current workflow step
                if db_step.step_number == current_step_number:
                    db_current_workflow_step = db_step

                if "project_id" in db_step.data:
                    db_step_project_id = db_step.data["project_id"]
                if "endpoint_name" in db_step.data:
                    db_step_endpoint_name = db_step.data["endpoint_name"]

            # Check duplicate endpoint in project
            query_endpoint_name = None
            query_project_id = None
            if project_id and db_step_endpoint_name:
                # If user gives project_id but endpoint_name given in earlier step
                query_endpoint_name = db_step_endpoint_name
                query_project_id = project_id
            elif endpoint_name and db_step_project_id:
                # If user gives endpoint_name but project_id given in earlier step
                query_endpoint_name = endpoint_name
                query_project_id = db_step_project_id
            elif endpoint_name and project_id:
                # if user gives both endpoint_name and project_id
                query_endpoint_name = endpoint_name
                query_project_id = project_id
            elif db_step_endpoint_name and db_step_project_id:
                # if user gives endpoint_name and project_id in earlier step
                query_endpoint_name = db_step_endpoint_name
                query_project_id = db_step_project_id

            if query_endpoint_name and query_project_id:
                # NOTE: A model can only be deployed once in a project
                db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
                    EndpointModel,
                    {
                        "name": query_endpoint_name,
                        "project_id": query_project_id,
                    },
                    exclude_fields={"status": EndpointStatusEnum.DELETED},
                    missing_ok=True,
                    case_sensitive=False,
                )
                if db_endpoint:
                    logger.info(
                        f"An endpoint with name {query_endpoint_name} already exists in project: {query_project_id}"
                    )
                    raise ClientException("An endpoint with this name already exists in this project")

        if db_current_workflow_step:
            logger.debug(f"Workflow {db_workflow.id} step {current_step_number} already exists")

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

        if deploy_config:
            # Fetch model information from previous steps
            db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
                {"workflow_id": db_workflow.id}
            )

            # Get latest model_id from workflow steps
            model_id = None
            for db_workflow_step in db_workflow_steps:
                if "model_id" in db_workflow_step.data:
                    model_id = db_workflow_step.data["model_id"]

            if not model_id:
                raise ClientException("Model ID is not provided")

            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model, {"id": model_id, "status": ModelStatusEnum.ACTIVE}
            )

            # Create next step from backend
            # Increment workflow_current_step with 1
            current_step_number = current_step_number + 1
            workflow_current_step = current_step_number

            bud_simulator_events = await self._perform_cluster_simulation(
                db_workflow.id,
                deploy_config,
                current_user_id,
                db_model.uri,
                db_model.provider_type,
                db_model.local_path,
            )
            simulator_id = bud_simulator_events.pop("workflow_id")
            recommended_cluster_events = {
                "simulator_id": simulator_id,
                "bud_simulator_events": bud_simulator_events,
            }

            # Update or create next workflow step
            db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
                db_workflow.id, current_step_number, recommended_cluster_events
            )
            logger.debug(f"Workflow step updated {db_workflow_step.id}")

            # Update workflow progress
            bud_simulator_events["progress_type"] = "bud_simulator_events"
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"progress": bud_simulator_events, "current_step": workflow_current_step},
            )

        # Execute workflow
        if trigger_workflow:
            logger.info("Workflow triggered")

            # Increment step number of workflow and workflow step
            current_step_number = current_step_number + 1
            workflow_current_step = current_step_number

            # TODO: Currently querying workflow steps again by ordering steps in ascending order
            # To ensure the latest step update is fetched, Consider excluding it later
            db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
                {"workflow_id": db_workflow.id}
            )

            # Define the keys required for model deployment
            keys_of_interest = [
                "model_id",
                "project_id",
                "cluster_id",
                # "created_by",
                "endpoint_name",
                "simulator_id",
                "deploy_config",
                "credential_id",
                "scaling_specification",
            ]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            required_keys = [
                "model_id",
                "project_id",
                "cluster_id",
                # "created_by",
                # "replicas",
                "endpoint_name",
                "simulator_id",
                "deploy_config",
                "scaling_specification",
            ]

            if "model_id" in required_data:
                db_model = await ModelDataManager(self.session).retrieve_by_fields(
                    Model,
                    {"id": required_data["model_id"], "status": ModelStatusEnum.ACTIVE},
                )

                if db_model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
                    required_keys.append("credential_id")

            # Check if all required keys are present
            missing_keys = [key for key in required_keys if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data: {', '.join(missing_keys)}")

            # Perform duplicate endpoint check
            db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
                EndpointModel,
                {
                    "name": required_data["endpoint_name"],
                    "project_id": required_data["project_id"],
                },
                exclude_fields={"status": EndpointStatusEnum.DELETED},
                missing_ok=True,
                case_sensitive=False,
            )

            if db_endpoint:
                raise ClientException("An endpoint with this name already exists in this project")

            # Trigger model deployment and get deployment events
            model_deployment_response = await self._initiate_model_deployment(
                cluster_id=UUID(required_data["cluster_id"]),
                endpoint_name=required_data["endpoint_name"],
                simulator_id=UUID(required_data["simulator_id"]),
                model_id=UUID(required_data["model_id"]),
                deploy_config=DeploymentTemplateCreate(**required_data["deploy_config"]),
                workflow_id=db_workflow.id,
                subscriber_id=current_user_id,
                credential_id=UUID(required_data["credential_id"]) if "credential_id" in required_data else None,
                scaling_specification=required_data["scaling_specification"],
            )
            model_deployment_events = {
                "budserve_cluster_events": model_deployment_response,
            }

            # Update or create next workflow step
            db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
                db_workflow.id, current_step_number, model_deployment_events
            )
            logger.debug(f"Workflow step updated {db_workflow_step.id}")

            # Update workflow progress
            model_deployment_response["progress_type"] = "budserve_cluster_events"
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"progress": model_deployment_response, "current_step": workflow_current_step},
            )
            logger.debug("Successfully triggered model deployment")

        return db_workflow

    @staticmethod
    async def _perform_cluster_simulation(
        workflow_id: UUID,
        deploy_config: DeploymentTemplateCreate,
        subscriber_id: UUID,
        model_uri: str,
        provider_type: ModelProviderTypeEnum,
        local_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get recommended cluster events."""
        logger.info("Getting recommended cluster events")

        notification_metadata = BudNotificationMetadata(
            workflow_id=str(workflow_id),
            subscriber_ids=str(subscriber_id),
            name=BUD_INTERNAL_WORKFLOW,
        )

        if provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
            recommended_cluster_request = RecommendedClusterRequest(
                pretrained_model_uri=model_uri,
                input_tokens=deploy_config.avg_context_length,
                output_tokens=deploy_config.avg_sequence_length,
                concurrency=deploy_config.concurrent_requests,
                target_ttft=0,
                target_throughput_per_user=0,
                target_e2e_latency=0,
                notification_metadata=notification_metadata,
                source_topic=app_settings.source_topic,
                is_proprietary_model=True,
            )
        else:
            # Only applicable for local model deployment
            # ttft and e2e latency minimum values
            # throughput maximum value
            target_throughput_per_user_max = (
                deploy_config.per_session_tokens_per_sec[1] if deploy_config.per_session_tokens_per_sec else None
            )
            ttft_min = deploy_config.ttft[0] if deploy_config.ttft else None
            e2e_latency_min = deploy_config.e2e_latency[0] if deploy_config.e2e_latency else None
            recommended_cluster_request = RecommendedClusterRequest(
                pretrained_model_uri=local_path,
                # pretrained_model_uri=model_uri, # Uncomment this for model uri
                input_tokens=deploy_config.avg_context_length,
                output_tokens=deploy_config.avg_sequence_length,
                concurrency=deploy_config.concurrent_requests,
                target_ttft=ttft_min,
                target_throughput_per_user=target_throughput_per_user_max,
                target_e2e_latency=e2e_latency_min,
                notification_metadata=notification_metadata,
                source_topic=app_settings.source_topic,
                is_proprietary_model=False,
            )

        # Get recommended cluster info from Bud Simulator
        recommended_cluster_endpoint = (
            f"{app_settings.dapr_base_url}v1.0/invoke/{app_settings.bud_simulator_app_id}/method/simulator/run"
        )

        # Perform recommended cluster simulation
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    recommended_cluster_endpoint, json=recommended_cluster_request.model_dump()
                ) as response:
                    response_data = await response.json()
                    if response.status >= 400:
                        raise ClientException("Unable to perform recommended cluster simulation")

                    # Add payload key to steps
                    if "steps" in response_data:
                        steps = response_data["steps"]
                        for step in steps:
                            step["payload"] = {}
                        response_data["steps"] = steps

                    return response_data
        except ClientException as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to perform recommended cluster simulation: {e}")
            raise ClientException("Unable to perform recommended cluster simulation") from e

    async def _initiate_model_deployment(
        self,
        cluster_id: UUID,
        endpoint_name: str,
        simulator_id: UUID,
        model_id: UUID,
        deploy_config: DeploymentTemplateCreate,
        workflow_id: UUID,
        subscriber_id: UUID,
        credential_id: UUID | None = None,
        scaling_specification: Optional[ScalingSpecification] = None,
    ) -> Dict[str, Any]:
        """Trigger model deployment by step."""
        logger.debug("Triggering model deployment")
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            Model, {"id": model_id, "status": ModelStatusEnum.ACTIVE}
        )
        logger.debug(f"Found model: {db_model.source}")

        target_throughput_per_user_max = (
            deploy_config.per_session_tokens_per_sec[1] if deploy_config.per_session_tokens_per_sec else None
        )
        ttft_min = deploy_config.ttft[0] if deploy_config.ttft else None
        e2e_latency_min = deploy_config.e2e_latency[0] if deploy_config.e2e_latency else None

        notification_metadata = BudNotificationMetadata(
            workflow_id=str(workflow_id),
            subscriber_ids=str(subscriber_id),
            name=BUD_INTERNAL_WORKFLOW,
        )

        if db_model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
            model_uri = db_model.uri
            model_source = db_model.source
            if model_uri.startswith(f"{model_source}/"):
                model_uri = model_uri.removeprefix(f"{model_source}/")
            deploy_model_uri = model_uri if not credential_id else f"{model_source}/{model_uri}"
            # Update made in lite-llm pr merge
        else:
            deploy_model_uri = db_model.local_path
            # deploy_model_uri = db_model.uri # Uncomment this for model uri

        # Perform model deployment
        model_deployment_request = ModelDeploymentRequest(
            cluster_id=cluster_id,
            simulator_id=simulator_id,
            endpoint_name=endpoint_name,
            model=deploy_model_uri,
            target_ttft=ttft_min,
            target_e2e_latency=e2e_latency_min,
            target_throughput_per_user=target_throughput_per_user_max,
            concurrency=deploy_config.concurrent_requests,
            input_tokens=deploy_config.avg_context_length,
            output_tokens=deploy_config.avg_sequence_length,
            notification_metadata=notification_metadata,
            source_topic=app_settings.source_topic,
            credential_id=credential_id,
            podscaler=scaling_specification,
        )
        model_deployment_endpoint = (
            f"{app_settings.dapr_base_url}v1.0/invoke/{app_settings.bud_cluster_app_id}/method/deployment"
        )
        model_deployment_payload = model_deployment_request.model_dump(mode="json", exclude_none=True)
        logger.debug("model_deployment_payload: %s", model_deployment_payload)

        # Perform model deployment
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(model_deployment_endpoint, json=model_deployment_payload) as response:
                    response_data = await response.json()
                    logger.debug("model_deployment_response: %s", response_data)

                    if response.status >= 400:
                        raise ClientException("Unable to perform model deployment")

                    # Add payload key to steps
                    if "steps" in response_data:
                        steps = response_data["steps"]
                        for step in steps:
                            step["payload"] = {}
                        response_data["steps"] = steps

                    return response_data
        except ClientException as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to perform model deployment: {e}")
            raise ClientException("Unable to perform model deployment") from e


class ModelServiceUtil(SessionMixin):
    """Model util service."""

    async def get_model_icon(self, db_model: Optional[Model] = None, model_id: Optional[UUID] = None) -> Optional[str]:
        """Get model icon.

        Args:
            db_model: The model to get the icon for.

        Returns:
            The model icon.
        """
        if db_model is None and model_id is None:
            raise ValueError("Atleast one of model instance or model id must be provided")
        if db_model is None:
            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model, {"id": model_id, "status": ModelStatusEnum.ACTIVE}
            )
        if db_model.provider_type in [ModelProviderTypeEnum.CLOUD_MODEL, ModelProviderTypeEnum.HUGGING_FACE]:
            return db_model.provider.icon
        else:
            return db_model.icon
