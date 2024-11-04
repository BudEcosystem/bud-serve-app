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

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import status

from budapp.commons import logging
from budapp.commons.constants import WorkflowStatusEnum
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.core.crud import WorkflowDataManager, WorkflowStepDataManager
from budapp.core.models import Workflow as WorkflowModel
from budapp.core.models import WorkflowStep as WorkflowStepModel

from .crud import CloudModelDataManager, ModelDataManager, ProviderDataManager
from .models import CloudModel, Model
from .models import Provider as ProviderModel
from .schemas import (
    CreateCloudModelWorkflowRequest,
    CreateCloudModelWorkflowResponse,
    CreateCloudModelWorkflowStepData,
    CreateCloudModelWorkflowSteps,
    EditModel,
    ModelCreate,
)
from pydantic import ValidationError



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

            # Trigger deploy model by step
            db_model = await self._execute_add_cloud_model_workflow(required_data, db_workflow.id)
            logger.debug(f"Successfully created model {db_model.id}")

        return db_workflow

    async def edit_cloud_model(self, model_id: UUID, data: Dict[str, Any]) -> None:
        """Edit cloud model by validating and updating specific fields."""

        # Retrieve the existing model from the database
        model = await ModelDataManager(self.session).get_model_by_id(model_id=model_id)
        if not model:
            raise ValueError(f"Model with ID {model_id} not found")

        # Validate and update fields using EditModel
        try:
            # Convert existing model to a dictionary of attributes
            model_data = {key: getattr(model, key) for key in model.__table__.columns.keys()}

            # Combine existing data with new update data
            updated_data = {**model_data, **data}

            # Validate data with EditModel schema
            validated_data = EditModel(**updated_data).dict(exclude_unset=True)

        except ValidationError as e:
            raise ValueError(f"Validation error: {e}")

        # Call update_model_by_fields with the validated data
        return await ModelDataManager(self.session).update_model_by_fields(model, validated_data)

        
    async def _execute_add_cloud_model_workflow(self, data: Dict[str, Any], workflow_id: UUID) -> None:
        """Execute add cloud model workflow."""
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Latest step
        db_latest_workflow_step = db_workflow_steps[-1]

        # Prepare model creation data from input
        model_data = await self._prepare_model_data(data)

        # Check for duplicate model
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            Model, {"uri": model_data.uri, "source": model_data.source}, missing_ok=True
        )
        if db_model:
            logger.info(f"Model {model_data.uri} with {model_data.source} already exists")
            raise ClientException("Duplicate model uri and source found")

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
        else:
            execution_status_data["model_id"] = str(db_model.id)
            db_workflow_step = await WorkflowStepDataManager(self.session).update_by_fields(
                db_latest_workflow_step, {"data": execution_status_data}
            )

            # leaderboard data. TODO: Need to add service for leaderboard
            leaderboard_data = await self._get_leaderboard_data()

            # Add leader board details
            end_step_number = db_workflow_step.step_number + 1

            # Create or update next workflow step
            db_workflow_step = await self._create_or_update_next_workflow_step(
                workflow_id, end_step_number, {"leaderboard": leaderboard_data}
            )

            # Update workflow current step
            db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(
                WorkflowModel, {"id": workflow_id}
            )
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"current_step": end_step_number},
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

    async def _prepare_model_data(self, data: Dict[str, Any]) -> None:
        """Prepare model data."""
        cloud_model_id = data.get("cloud_model_id")
        source = data.get("source")
        name = data.get("name")
        modality = data.get("modality")
        uri = data.get("uri")
        tags = data.get("tags")
        provider_type = data.get("provider_type")
        provider_id = data.get("provider_id")
        created_by = data.get("created_by")

        db_provider = await ProviderDataManager(self.session).retrieve_by_fields(
            ProviderModel, {"id": provider_id}, missing_ok=True
        )
        icon = db_provider.icon

        if cloud_model_id:
            db_cloud_model = await CloudModelDataManager(self.session).retrieve_by_fields(
                CloudModel, {"id": cloud_model_id}, missing_ok=True
            )
            model_data = ModelCreate(
                name=name,
                description=db_cloud_model.description,
                tags=tags,
                tasks=db_cloud_model.tasks,
                author=db_cloud_model.author,
                model_size=db_cloud_model.model_size,
                icon=icon,
                github_url=db_cloud_model.github_url,
                huggingface_url=db_cloud_model.huggingface_url,
                website_url=db_cloud_model.website_url,
                modality=db_cloud_model.modality,
                # type=db_cloud_model.type,
                source=db_cloud_model.source,
                provider_type=provider_type,
                uri=db_cloud_model.uri,
                created_by=UUID(created_by),
            )
        else:
            model_data = ModelCreate(
                source=source,
                name=name,
                modality=modality,
                uri=uri,
                tags=tags,
                provider_type=provider_type,
                icon=icon,
                created_by=UUID(created_by),
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
        return await CloudModelDataManager(self.session).get_all_cloud_models(offset, limit, filters, order_by, search)

    async def get_all_recommended_tags(
        self,
        offset: int = 0,
        limit: int = 10,
    ) -> Tuple[List[CloudModel], int]:
        """Get all cloud models."""
        return await CloudModelDataManager(self.session).get_all_recommended_tags(offset, limit)
    
class ModelService(SessionMixin):
    """Cloud model service."""
    
    async def get_model_details(self, model_id: UUID) -> Model:
        """Retrieve model details by model ID."""
        return await ModelDataManager(self.session).get_model_by_id(model_id=model_id)

