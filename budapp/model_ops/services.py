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
from budapp.commons.constants import WorkflowStatusEnum
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.commons.schemas import Tag, Task
from budapp.workflow_ops.crud import WorkflowDataManager, WorkflowStepDataManager
from budapp.workflow_ops.models import Workflow as WorkflowModel
from budapp.workflow_ops.models import WorkflowStep as WorkflowStepModel

from .crud import CloudModelDataManager, ModelDataManager, ProviderDataManager
from .models import CloudModel, Model, ModelLicenses, PaperPublished
from .models import Provider as ProviderModel
from .schemas import (
    CreateCloudModelWorkflowRequest,
    CreateCloudModelWorkflowResponse,
    CreateCloudModelWorkflowStepData,
    CreateCloudModelWorkflowSteps,
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

    async def edit_cloud_model(self, model_id: UUID, data: Dict[str, Any], file: UploadFile = None) -> None:
        """Edit cloud model by validating and updating specific fields, and saving an uploaded file if provided."""
        # Retrieve existing model
        model = await ModelDataManager(self.session).retrieve_by_fields(model=Model, fields={"id": model_id})
        if not model:
            raise ValueError(f"Model with ID {model_id} not found")

        # Handle file upload if provided
        if file:
            file_path = await self._save_uploaded_file(file)
            await self._create_or_update_license_entry(model_id, file.filename, file_path)

        # Handle license URL if provided
        elif data.get("license_url"):
            file_path, filename = await self._download_license_file(data.pop("license_url"))
            await self._create_or_update_license_entry(model_id, filename, file_path)

        # Add papers if provided
        if data.get("paper_urls"):
            await self._add_papers(model_id, data.pop("paper_urls"))

        # Validate and update fields
        validated_data = await self._validate_update_data(model, data)

        # Update model with validated data
        await ModelDataManager(self.session).update_by_fields(model, validated_data)

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

    async def _validate_update_data(self, model: Model, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and prepare update data using EditModel schema."""
        model_data = {key: getattr(model, key) for key in model.__table__.columns.keys()}
        updated_data = {**model_data, **data}
        try:
            return EditModel(**updated_data).dict(exclude_unset=True, exclude_none=True)
        except ValidationError as e:
            raise ValueError(f"Validation error: {e}")

    async def _add_papers(self, model_id: UUID, paper_urls: list[str]) -> None:
        """Add paper entries if paper URLs are provided."""
        for paper_url in paper_urls:
            paper_entry = PaperPublishedModel(id=uuid4(), title=None, url=paper_url, model_id=model_id)
            await ModelDataManager(self.session).insert_one(
                PaperPublished(**paper_entry.model_dump(exclude_unset=True))
            )

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
