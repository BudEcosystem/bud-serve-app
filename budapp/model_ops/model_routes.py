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

"""The model ops package, containing essential business logic, services, and routing configurations for the model ops."""

import json
from typing import List, Optional, Union
from uuid import UUID

from fastapi import Form, File, UploadFile
from fastapi import APIRouter, Depends, Query, status
from pydantic import ValidationError
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons import logging
from budapp.commons.dependencies import (
    get_current_active_user,
    get_session,
    parse_ordering_fields,
)
from budapp.commons.exceptions import ClientException
from budapp.commons.schemas import ErrorResponse, SuccessResponse
from budapp.user_ops.schemas import User

from .schemas import (
    CloudModelFilter,
    CloudModelResponse,
    CreateCloudModelWorkflowRequest,
    CreateCloudModelWorkflowResponse,
    EditModel,
    ProviderFilter,
    ProviderResponse,
    RecommendedTagsResponse,
)
from .services import CloudModelService, CloudModelWorkflowService, ProviderService


logger = logging.get_logger(__name__)

model_router = APIRouter(prefix="/models", tags=["model"])


@model_router.get(
    "/providers",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
        status.HTTP_200_OK: {
            "model": ProviderResponse,
            "description": "Successfully list all providers",
        },
    },
    description="List all model providers",
)
async def list_providers(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    filters: ProviderFilter = Depends(),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),
    search: bool = False,
) -> Union[ProviderResponse, ErrorResponse]:
    """List all model providers."""
    # Calculate offset
    offset = (page - 1) * limit

    # Convert UserFilter to dictionary
    filters_dict = filters.model_dump(exclude_none=True)

    try:
        db_providers, count = await ProviderService(session).get_all_providers(
            offset, limit, filters_dict, order_by, search
        )
    except Exception as e:
        logger.exception(f"Failed to get all providers: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get all providers"
        ).to_http_response()

    return ProviderResponse(
        providers=db_providers,
        total_record=count,
        page=page,
        limit=limit,
        object="providers.list",
        code=status.HTTP_200_OK,
    ).to_http_response()


@model_router.post(
    "/cloud-model-workflow",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
        status.HTTP_200_OK: {
            "model": CreateCloudModelWorkflowResponse,
            "description": "Successfully add cloud model workflow",
        },
    },
    description="Add cloud model workflow",
)
async def add_cloud_model_workflow(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    request: CreateCloudModelWorkflowRequest,
) -> Union[CreateCloudModelWorkflowResponse, ErrorResponse]:
    """Add cloud model workflow."""
    try:
        db_workflow = await CloudModelWorkflowService(session).add_cloud_model_workflow(
            current_user_id=current_user.id,
            request=request,
        )

        return await CloudModelWorkflowService(session).get_cloud_model_workflow(db_workflow.id)
    except ClientException as e:
        logger.exception(f"Failed to add cloud model workflow: {e}")
        return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to add cloud model workflow: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to add cloud model workflow"
        ).to_http_response()


@model_router.patch(
    "/{model_id}",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
        status.HTTP_200_OK: {
            "model": CreateCloudModelWorkflowResponse,
            "description": "Successfully edited cloud model",
        },
    },
    description="Edit cloud model",
)
async def edit_model(
    model_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    name: Optional[str] = Form(None, min_length=1, max_length=100),
    description: Optional[str] = Form(None, max_length=500),
    tags: Optional[str] = Form(None),  # JSON string of tags
    tasks: Optional[str] = Form(None),  # JSON string of tasks
    paper_urls: Optional[str] = Form(None),
    github_url: Optional[str] = Form(None),
    huggingface_url: Optional[str] = Form(None),
    website_url: Optional[str] = Form(None),
    license_file: Optional[UploadFile] = File(None),
    license_url: Optional[str] = Form(None)
) -> Union[SuccessResponse, ErrorResponse]:
    """Edit cloud model with file upload"""
    try:
        # Parse JSON strings for list fields
        tags = json.loads(tags) if tags else None
        tasks = json.loads(tasks) if tasks else None
        paper_urls = json.loads(paper_urls) if paper_urls else None

        try:
            # Convert to EditModel
            edit_model = EditModel(
                name=name,
                description=description,
                tags=tags,
                tasks=tasks,
                paper_urls=paper_urls,
                github_url=github_url,
                huggingface_url=huggingface_url,
                website_url=website_url,
                license_url=license_url
            )
        except ValidationError as e:
            logger.exception(f"Failed to edit cloud model: {e}")
            return ErrorResponse(code=status.HTTP_422_UNPROCESSABLE_ENTITY, message='Validation error').to_http_response()
        
        # Pass file and edit_model data to your service
        await CloudModelWorkflowService(session).edit_cloud_model(
            model_id=model_id,
            data=edit_model.dict(exclude_unset=True),
            file=license_file
        )

        return SuccessResponse(message="Cloud model edited successfully", code=status.HTTP_200_OK).to_http_response()
    except ClientException as e:
        logger.exception(f"Failed to edit cloud model: {e}")
        return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to edit cloud model: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to edit cloud model"
        ).to_http_response()


@model_router.get(
    "/cloud-model-workflow/{workflow_id}",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
        status.HTTP_200_OK: {
            "model": CreateCloudModelWorkflowResponse,
            "description": "Successfully add cloud model workflow",
        },
    },
    description="Get cloud model workflow",
)
async def get_cloud_model_workflow(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    workflow_id: UUID,
) -> Union[CreateCloudModelWorkflowResponse, ErrorResponse]:
    """Get cloud model workflow."""
    try:
        return await CloudModelWorkflowService(session).get_cloud_model_workflow(workflow_id)
    except ClientException as e:
        logger.exception(f"Failed to get cloud model workflow: {e}")
        return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get cloud model workflow: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get cloud model workflow"
        ).to_http_response()


@model_router.get(
    "/cloud-models",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
        status.HTTP_200_OK: {
            "model": ProviderResponse,
            "description": "Successfully list all providers",
        },
    },
    description="List all cloud models",
)
async def list_cloud_models(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    filters: Annotated[CloudModelFilter, Depends()],
    tags: List[str] = Query(default_factory=list),
    tasks: List[str] = Query(default_factory=list),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),
    search: bool = False,
) -> Union[CloudModelResponse, ErrorResponse]:
    """List all cloud models."""
    # Calculate offset
    offset = (page - 1) * limit

    # Convert UserFilter to dictionary
    filters_dict = filters.model_dump(exclude_none=True)
    if tags:
        filters_dict["tags"] = tags
    if tasks:
        filters_dict["tasks"] = tasks

    try:
        db_models, count = await CloudModelService(session).get_all_cloud_models(
            offset, limit, filters_dict, order_by, search
        )
    except Exception as e:
        logger.exception(f"Failed to get all cloud models: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get all cloud models"
        ).to_http_response()

    return CloudModelResponse(
        cloud_models=db_models,
        total_record=count,
        page=page,
        limit=limit,
        object="cloud_models.list",
        code=status.HTTP_200_OK,
    ).to_http_response()


@model_router.get(
    "/cloud-models/recommended-tags",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
        status.HTTP_200_OK: {
            "model": RecommendedTagsResponse,
            "description": "Successfully list all recommended tags",
        },
    },
    description="List all cloud model recommended tags",
)
async def list_cloud_model_recommended_tags(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
) -> Union[RecommendedTagsResponse, ErrorResponse]:
    """List all most used tags."""
    # Calculate offset
    offset = (page - 1) * limit

    try:
        db_tags, count = await CloudModelService(session).get_all_recommended_tags(offset, limit)
    except Exception as e:
        logger.exception(f"Failed to get all recommended tags: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get all recommended tags"
        ).to_http_response()

    return RecommendedTagsResponse(
        tags=db_tags,
        total_record=count,
        page=page,
        limit=limit,
        object="recommended_tags.list",
        code=status.HTTP_200_OK,
    ).to_http_response()
