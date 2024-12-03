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
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import aiofiles
import aiohttp
from fastapi import UploadFile, status
from pydantic import ValidationError

from budapp.commons import logging
from budapp.commons.config import app_settings
from budapp.commons.constants import (
    BudServeWorkflowStepEventName,
    CredentialTypeEnum,
    ModelProviderTypeEnum,
    ModelSourceEnum,
    WorkflowStatusEnum,
    ModelProviderTypeEnum,
)
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.commons.helpers import assign_random_colors_to_names, get_normalized_string_or_none
from budapp.commons.schemas import Tag, Task
from budapp.core.schemas import NotificationPayload
from budapp.credential_ops.crud import ProprietaryCredentialDataManager
from budapp.credential_ops.models import ProprietaryCredential as ProprietaryCredentialModel
from budapp.workflow_ops.crud import WorkflowDataManager, WorkflowStepDataManager
from budapp.workflow_ops.models import Workflow as WorkflowModel
from budapp.workflow_ops.models import WorkflowStep as WorkflowStepModel
from budapp.workflow_ops.services import WorkflowService, WorkflowStepService

from .crud import CloudModelDataManager, ModelDataManager, ProviderDataManager
from .models import CloudModel, Model, ModelLicenses, PaperPublished
from .models import Provider as ProviderModel
from .schemas import (
    CreateCloudModelWorkflowRequest,
    CreateCloudModelWorkflowResponse,
    CreateCloudModelWorkflowStepData,
    CreateCloudModelWorkflowSteps,
    CreateLocalModelWorkflowRequest,
    CreateLocalModelWorkflowSteps,
    EditModel,
    ModelCreate,
    ModelLicensesModel,
    ModelListResponse,
    ModelResponse,
    PaperPublishedModel,
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

        current_step_number = step_number

        # Retrieve or create workflow
        db_workflow = await self._retrieve_or_create_workflow(workflow_id, workflow_total_steps, current_user_id)

        # Model source is provider type
        source = None
        if provider_id:
            db_provider = await ProviderDataManager(self.session).retrieve_by_fields(
                ProviderModel, {"id": provider_id}
            )
            source = db_provider.type.value

        if cloud_model_id:
            db_cloud_model = await CloudModelDataManager(self.session).retrieve_by_fields(
                CloudModel, {"id": cloud_model_id}
            )

            if db_cloud_model.is_present_in_model:
                raise ClientException("Cloud model is already present in model")

        if name:
            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model, {"name": name, "is_active": True}, missing_ok=True
            )
            if db_model:
                raise ClientException("Model name already exists")

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
                "created_by",
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
                Model, {"name": required_data["name"], "is_active": True}, missing_ok=True
            )
            if db_model:
                raise ClientException("Model name already exists")

            # Trigger deploy model by step
            db_model = await self._execute_add_cloud_model_workflow(required_data, db_workflow.id)
            logger.debug(f"Successfully created model {db_model.id}")

        return db_workflow

    async def _execute_add_cloud_model_workflow(self, data: Dict[str, Any], workflow_id: UUID) -> None:
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
                CloudModel, {"id": cloud_model_id}, missing_ok=True
            )

        # Prepare model creation data from input
        model_data = await self._prepare_model_data(data, db_cloud_model)

        # Check for duplicate model
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            Model, {"uri": model_data.uri, "source": model_data.source}, missing_ok=True
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
        return db_model

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
                    "is_active": True,
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

    async def _prepare_model_data(self, data: Dict[str, Any], db_cloud_model: Optional[CloudModel] = None) -> None:
        """Prepare model data."""
        source = data.get("source")
        name = data.get("name")
        modality = data.get("modality")
        uri = data.get("uri")
        tags = data.get("tags")
        provider_type = data.get("provider_type")
        provider_id = data.get("provider_id")
        created_by = data.get("created_by")

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
                created_by=UUID(created_by),
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
                created_by=UUID(created_by),
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

        current_step_number = step_number

        # Retrieve or create workflow
        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id, workflow_total_steps, current_user_id
        )

        # Validate proprietary credential id
        if proprietary_credential_id:
            await ProprietaryCredentialDataManager(self.session).retrieve_by_fields(
                ProprietaryCredentialModel, {"id": proprietary_credential_id}
            )

        # Validate model name to be unique
        if name:
            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model, {"name": name, "is_active": True}, missing_ok=True
            )
            if db_model:
                raise ClientException("Model name should be unique")

        # Add provider_id for HuggingFace provider type
        provider_id = None
        if provider_type == ModelProviderTypeEnum.HUGGING_FACE:
            db_provider = await ProviderDataManager(self.session).retrieve_by_fields(
                ProviderModel, {"type": CredentialTypeEnum.HUGGINGFACE}
            )
            provider_id = db_provider.id

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
        ).model_dump(exclude_none=True, exclude_unset=True, mode="json")

        # Get workflow steps
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": db_workflow.id}
        )

        # For avoiding another db call for record retrieval, storing db object while iterating over db_workflow_steps
        db_current_workflow_step = None

        # Verify hugging face uri duplication
        await self._verify_hugging_face_uri_duplication(provider_type, uri, db_workflow_steps)

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
                "created_by",
            ]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            # Check if all required keys are present
            required_keys = ["provider_type", "name", "uri", "created_by"]
            missing_keys = [key for key in required_keys if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data for model extraction: {', '.join(missing_keys)}")

            # Create or update new workflow step for model extraction
            current_step_number = current_step_number + 1

            try:
                # Perform model extraction
                await self._perform_model_extraction(db_workflow.id, current_step_number, required_data)
            except ClientException as e:
                workflow_current_step = current_step_number
                db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                    db_workflow,
                    {"current_step": workflow_current_step},
                )
                logger.debug("Workflow updated with latest step")
                raise e

            # Create next workflow step to store model extraction response
            current_step_number = current_step_number + 1
            workflow_current_step = current_step_number

            # Update or create next workflow step
            # NOTE: The when extraction is done, the subscriber will create model and update model_id to the step
            db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
                workflow_id, current_step_number, {}
            )

        # This will ensure workflow step number is updated to the latest step number
        db_workflow = await WorkflowDataManager(self.session).update_by_fields(
            db_workflow,
            {"current_step": workflow_current_step},
        )

        return db_workflow

    async def create_model_from_notification_event(self, payload: NotificationPayload) -> None:
        """Create a local model from notification event."""
        logger.debug("Received event for creating local model")

        # Get workflow steps
        workflow_id = payload.workflow_id
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Get last step
        db_latest_workflow_step = db_workflow_steps[-1]

        # Define the keys required for endpoint creation
        keys_of_interest = [
            "provider_type",
            "name",
            "uri",
            "author",
            "tags",
            "icon",
            "created_by",
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
            Model, {"name": required_data["name"], "is_active": True}, missing_ok=True
        )
        if db_model:
            logger.error(f"Unable to create model with name {required_data['name']} as it already exists")
            raise ClientException("Model name should be unique")

        model_info = payload.content.result["model_info"]
        local_path = payload.content.result["local_path"]

        # Extract and finalize tags, tasks and author
        given_tags = required_data.get("tags", [])
        extracted_tags = model_info.get("tags", [])
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
            extracted_author = get_normalized_string_or_none(model_info.get("author", None))

        # Finalize model details
        model_description = get_normalized_string_or_none(model_info.get("description", None))
        model_github_url = get_normalized_string_or_none(model_info.get("huggingface_url", None))
        model_huggingface_url = get_normalized_string_or_none(model_info.get("huggingface_url", None))
        model_website_url = get_normalized_string_or_none(model_info.get("website_url", None))

        # Set provider id and icon
        provider_id = None
        icon = required_data.get("icon")
        if required_data["provider_type"] == ModelProviderTypeEnum.HUGGING_FACE.value:
            # icon is not supported for hugging face models
            # Add provider id for hugging face models to retrieve icon for frontend
            icon = None
            db_provider = await ProviderDataManager(self.session).retrieve_by_fields(
                ProviderModel, {"type": "huggingface"}
            )
            provider_id = db_provider.id

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
            created_by=UUID(required_data["created_by"]),
            author=extracted_author,
            provider_id=provider_id,
            local_path=local_path,
            icon=icon,
        )

        # Create model
        db_model = await ModelDataManager(self.session).insert_one(Model(**model_data.model_dump()))
        logger.debug(f"Model created with id {db_model.id}")

        # Update to workflow step
        workflow_update_data = {
            "model_id": str(db_model.id),
            "tags": extracted_tags,
            "description": model_description,
        }
        await WorkflowStepDataManager(self.session).update_by_fields(
            db_latest_workflow_step, {"data": workflow_update_data}
        )
        logger.debug(f"Workflow step updated with model id {db_model.id}")

        # Mark workflow as completed
        logger.debug(f"Updating workflow status: {workflow_id}")
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})
        await WorkflowDataManager(self.session).update_by_fields(db_workflow, {"status": WorkflowStatusEnum.COMPLETED})

    async def _verify_hugging_face_uri_duplication(
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
            # Check duplicate hugging face uri
            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model, {"uri": query_uri, "provider_type": query_provider_type, "is_active": True}, missing_ok=True
            )
            if db_model:
                raise ClientException("Duplicate hugging face uri found")

    async def _perform_model_extraction(self, workflow_id: UUID, step_number: int, data: Dict) -> None:
        """Perform model extraction."""
        # Update or create next workflow step
        db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
            workflow_id, step_number, {}
        )

        # Perform model extraction request
        model_extraction_response = await self._perform_model_extraction_request(workflow_id, data)

        # Add payload dict to response
        for step in model_extraction_response["steps"]:
            step["payload"] = {}

        model_extraction_events = {
            BudServeWorkflowStepEventName.MODEL_EXTRACTION_EVENTS.value: model_extraction_response
        }

        # Update workflow step with response
        await WorkflowStepDataManager(self.session).update_by_fields(
            db_workflow_step, {"data": model_extraction_events}
        )

    async def _perform_model_extraction_request(self, workflow_id: UUID, data: Dict) -> None:
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

            # TODO: remove this after implementing token decryption, decryption not required here
            # Send decrypted token to model extraction endpoint
            hf_token = await self._get_decrypted_token(proprietary_credential_id)

        model_extraction_request = {
            "model_name": data["name"],
            "model_uri": data["uri"],
            "provider_type": data["provider_type"],
            "hf_token": hf_token,
            "notification_metadata": {
                "name": "bud-notification",
                "subscriber_ids": data["created_by"],
                "workflow_id": str(workflow_id),
            },
            "source_topic": f"{app_settings.source_topic}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(model_extraction_endpoint, json=model_extraction_request) as response:
                    response_data = await response.json()
                    if response.status >= 400:
                        raise ClientException("unable to perform model extraction request")

                    return response_data
        except ClientException as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to perform model extraction request: {e}")
            raise ClientException("unable to perform model extraction request") from e

    @staticmethod
    async def _get_decrypted_token(credential_id: UUID) -> str:
        """Get decrypted token."""
        # TODO: remove this function after implementing dapr decryption
        url = f"https://api-dev.bud.studio/proprietary/credentials/{credential_id}/details"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response_data = await response.json()
                try:
                    decrypted_token = response_data["result"]["other_provider_creds"]["api_key"]
                    return decrypted_token
                except (KeyError, TypeError):
                    raise ClientException("Unable to get decrypted token")


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
    """Cloud model service."""

    async def get_faqs(self) -> List[Dict[str, Any]]:
        """Dummy function to return FAQs for the license."""
        return [
            {"answer": True, "question": "Are the weights of models are opensource?"},
            {"answer": False, "question": "Are the weights of models are opensource?"},
        ]

    async def get_model_details(self, model_id: UUID) -> Model:
        """Retrieve model details by model ID."""
        model_details = await ModelDataManager(self.session).retrieve_by_fields(
            Model, {"id": model_id}, missing_ok=True
        )

        if not model_details:
            raise ClientException(message="Model not found", status_code=404)

        paper_published_list = [PaperPublishedModel.from_orm(paper) for paper in model_details.paper_published]
        if model_details.model_licenses:
            license = ModelLicensesModel.from_orm(model_details.model_licenses)
            license_data = license.dict()
            license_data["faqs"] = await self.get_faqs()
        else:
            license_data = None

        response_data = {
            "id": model_details.id,
            "name": model_details.name,
            "description": model_details.description,
            "icon": model_details.icon,
            "tags": model_details.tags,
            "tasks": model_details.tasks,
            "github_url": model_details.github_url,
            "huggingface_url": model_details.huggingface_url,
            "website_url": model_details.website_url,
            "paper_published": paper_published_list,
            "license": license_data,
            "provider_type": model_details.provider_type,
            "provider": model_details.provider if model_details.provider else None,
        }
        return response_data

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
        filters["is_active"] = True
        db_models, count = await ModelDataManager(self.session).list_all_model_authors(
            offset, limit, filters, order_by, search
        )
        db_authors = [model.author for model in db_models]

        return db_authors, count

    async def edit_cloud_model(self, model_id: UUID, data: Dict[str, Any], file: UploadFile = None) -> None:
        """Edit cloud model by validating and updating specific fields, and saving an uploaded file if provided."""
        # Retrieve existing model
        db_model = await ModelDataManager(self.session).retrieve_by_fields(model=Model, fields={"id": model_id})
        if data.get("icon") and db_model.provider_type in [
            ModelProviderTypeEnum.CLOUD_MODEL,
            ModelProviderTypeEnum.HUGGING_FACE,
        ]:
            data.pop("icon")
        # Handle file upload if provided
        # TODO: consider dapr local storage
        if file:
            file_path = await self._save_uploaded_file(file)
            await self._create_or_update_license_entry(model_id, file.filename, file_path)

        # Handle license URL if provided
        elif data.get("license_url"):
            file_path, filename = await self._download_license_file(data.pop("license_url"))
            await self._create_or_update_license_entry(model_id, filename, file_path)

        # Add papers if provided
        if data.get("paper_urls") or data["paper_urls"] == []:
            await self._update_papers(model_id, data.pop("paper_urls"))

        updated_data = {
            key: data[key] if key in data and data[key] is not None else getattr(db_model, key)
            for key in db_model.__table__.columns.keys()
        }

        # Update model with validated data
        await ModelDataManager(self.session).update_by_fields(db_model, updated_data)

    async def _save_uploaded_file(self, file: UploadFile) -> str:
        """Save uploaded file and return file path."""
        file_directory = "static/licenses"
        os.makedirs(file_directory, exist_ok=True)
        file_path = os.path.join(file_directory, file.filename)
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        return file_path

    async def _download_license_file(self, license_url: str) -> tuple[str, str]:
        """Download file from URL and save locally, returning file path and name."""
        file_directory = "static/licenses"
        os.makedirs(file_directory, exist_ok=True)
        filename = license_url.split("/")[-1]
        file_path = os.path.join(file_directory, filename)
        async with aiohttp.ClientSession() as session:
            async with session.get(license_url) as response:
                if response.status == 200:
                    content = await response.read()
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(content)
                else:
                    raise ClientException("Failed to download the file from the provided URL")
        return file_path, filename

    async def _create_or_update_license_entry(self, model_id: UUID, filename: str, file_path: str) -> None:
        """Create or update a license entry in the database."""
        # Check if a license entry with the given model_id exists
        existing_license = await ModelDataManager(self.session).retrieve_by_fields(
            ModelLicenses, fields=dict(model_id=model_id), missing_ok=True
        )

        if existing_license:
            if existing_license.path and os.path.exists(existing_license.path):
                os.remove(existing_license.path)

            # Update the existing license entry
            existing_license.name = filename
            existing_license.path = file_path
            print(existing_license)
            ModelDataManager(self.session).update_one(existing_license)
        else:
            # Create a new license entry
            license_entry = ModelLicensesModel(id=uuid4(), name=filename, path=file_path, model_id=model_id)
            await ModelDataManager(self.session).insert_one(
                ModelLicenses(**license_entry.model_dump(exclude_unset=True))
            )

    async def _update_papers(self, model_id: UUID, paper_urls: list[str]) -> None:
        """Update paper entries for the given model by adding new URLs and removing old ones."""
        # Fetch existing paper URLs for the model
        existing_papers = await ModelDataManager(self.session).retrieve_all_by_fields(
            model=PaperPublished, fields={"model_id": model_id}, missing_ok=True
        )
        existing_urls = {paper.url for paper in existing_papers}

        # Determine URLs to add and remove
        input_urls = set(paper_urls)
        urls_to_add = input_urls - existing_urls
        urls_to_remove = existing_urls - input_urls
        logger.debug(f"paper info: {input_urls}, urls_to_add: {urls_to_add}, urls_to_remove: {urls_to_remove}")

        # Add new paper URLs
        for paper_url in urls_to_add:
            paper_entry = PaperPublishedModel(id=uuid4(), title=None, url=paper_url, model_id=model_id)
            await ModelDataManager(self.session).insert_one(
                PaperPublished(**paper_entry.model_dump(exclude_unset=True))
            )

        # Remove old paper URLs
        if urls_to_remove:
            for paper_url in urls_to_remove:
                # Iterate through the URLs to remove and delete each matching entry by 'url' and 'model_id'
                await ModelDataManager(self.session).delete_by_fields(
                    model=PaperPublished, fields={"url": paper_url, "model_id": model_id}
                )
