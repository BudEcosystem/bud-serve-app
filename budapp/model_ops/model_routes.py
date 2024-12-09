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
from json.decoder import JSONDecodeError
from typing import List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, Form, Query, UploadFile, status
from pydantic import ValidationError
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons import logging
from budapp.commons.constants import ModalityEnum
from budapp.commons.dependencies import (
    get_current_active_user,
    get_session,
    parse_ordering_fields,
)
from budapp.commons.exceptions import ClientException
from budapp.commons.schemas import ErrorResponse, SuccessResponse
from budapp.user_ops.schemas import User
from budapp.workflow_ops.schemas import RetrieveWorkflowDataResponse
from budapp.workflow_ops.services import WorkflowService

from .schemas import (
    CreateCloudModelWorkflowRequest,
    CreateCloudModelWorkflowResponse,
    CreateLocalModelWorkflowRequest,
    EditModel,
    ModelAuthorFilter,
    ModelAuthorResponse,
    ModelDetailSuccessResponse,
    ModelFilter,
    ModelPaginatedResponse,
    ProviderFilter,
    ProviderResponse,
    RecommendedTagsResponse,
    TagsListResponse,
    TasksListResponse,
)
from .services import (
    CloudModelService,
    CloudModelWorkflowService,
    LocalModelWorkflowService,
    ModelService,
    ProviderService,
)


logger = logging.get_logger(__name__)

model_router = APIRouter(prefix="/models", tags=["model"])


@model_router.get(
    "/",
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
            "model": ModelPaginatedResponse,
            "description": "Successfully list all models",
        },
    },
    description="List all models",
)
async def list_all_models(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    filters: Annotated[ModelFilter, Depends()],
    author: List[str] = Query(default=[]),
    modality: List[ModalityEnum] = Query(default=[]),
    tags: List[str] = Query(default=[]),
    tasks: List[str] = Query(default=[]),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),
    search: bool = False,
) -> Union[ModelPaginatedResponse, ErrorResponse]:
    """List all models."""
    # Calculate offset
    offset = (page - 1) * limit

    # Convert UserFilter to dictionary
    filters_dict = filters.model_dump(exclude_none=True, exclude={"table_source"})

    # Update filters_dict only for non-empty lists
    filter_updates = {"tags": tags, "tasks": tasks, "author": author, "modality": modality}
    filters_dict.update({k: v for k, v in filter_updates.items() if v})

    # Perform router level validation
    if filters.table_source == "cloud_model" and filter_updates["author"]:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            message="Author is not allowed for cloud models.",
        ).to_http_response()

    if filters.model_size_min and filters.model_size_max and filters.model_size_min > filters.model_size_max:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            message="Model size min is greater than model size max.",
        ).to_http_response()

    try:
        if filters.table_source == "cloud_model":
            db_models, count = await CloudModelService(session).get_all_cloud_models(
                offset, limit, filters_dict, order_by, search
            )
        else:
            db_models, count = await ModelService(session).get_all_active_models(
                offset, limit, filters_dict, order_by, search
            )
    except ClientException as e:
        logger.exception(f"Failed to get all models: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get all models: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get all cloud models"
        ).to_http_response()

    return ModelPaginatedResponse(
        models=db_models,
        total_record=count,
        page=page,
        limit=limit,
        object="models.list",
        code=status.HTTP_200_OK,
    ).to_http_response()


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
            "model": RetrieveWorkflowDataResponse,
            "description": "Successfully add cloud model workflow",
        },
    },
    description="Add cloud model workflow",
)
async def add_cloud_model_workflow(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    request: CreateCloudModelWorkflowRequest,
) -> Union[RetrieveWorkflowDataResponse, ErrorResponse]:
    """Add cloud model workflow."""
    try:
        db_workflow = await CloudModelWorkflowService(session).add_cloud_model_workflow(
            current_user_id=current_user.id,
            request=request,
        )

        return await WorkflowService(session).retrieve_workflow_data(db_workflow.id)
    except ClientException as e:
        logger.exception(f"Failed to add cloud model workflow: {e}")
        return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to add cloud model workflow: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to add cloud model workflow"
        ).to_http_response()


@model_router.post(
    "/local-model-workflow",
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
            "model": RetrieveWorkflowDataResponse,
            "description": "Successfully add local model workflow",
        },
    },
    description="Add local model workflow",
)
async def add_local_model_workflow(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    request: CreateLocalModelWorkflowRequest,
) -> Union[RetrieveWorkflowDataResponse, ErrorResponse]:
    """Add local model workflow."""
    try:
        db_workflow = await LocalModelWorkflowService(session).add_local_model_workflow(
            current_user_id=current_user.id,
            request=request,
        )

        return await WorkflowService(session).retrieve_workflow_data(db_workflow.id)
    except ClientException as e:
        logger.exception(f"Failed to add local model workflow: {e}")
        return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to add local model workflow: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to add local model workflow"
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
    paper_urls: Optional[list[str]] = Form(None),
    github_url: Optional[str] = Form(None),
    huggingface_url: Optional[str] = Form(None),
    website_url: Optional[str] = Form(None),
    license_file: UploadFile | None = None,
    license_url: Optional[str] = Form(None),
) -> Union[SuccessResponse, ErrorResponse]:
    """Edit cloud model with file upload"""
    logger.info(
        f"Received data: name={name}, description={description}, tags={tags}, tasks={tasks}, paper_urls={paper_urls}, github_url={github_url}, huggingface_url={huggingface_url}, website_url={website_url}, license_file={license_file}, license_url={license_url}"
    )
    try:
        # Parse JSON strings for list fields
        tags = json.loads(tags) if tags else None
        tasks = json.loads(tasks) if tasks else None

        # Convert to list of multiple strings from a list of single string with comma separated values
        if paper_urls and isinstance(paper_urls, list) and len(paper_urls) > 0:
            paper_urls = [url.strip() for url in paper_urls[0].split(",")]

        try:
            # Convert to EditModel
            edit_model = EditModel(
                name=name if name else None,
                description=description if description else None,
                tags=tags if tags else None,
                tasks=tasks if tasks else None,
                paper_urls=paper_urls if paper_urls else None,
                github_url=github_url if github_url else None,
                huggingface_url=huggingface_url if huggingface_url else None,
                website_url=website_url if website_url else None,
                license_url=license_url if license_url else None,
            )
        except ValidationError as e:
            logger.exception(f"Failed to edit cloud model: {e}")
            return ErrorResponse(
                code=status.HTTP_422_UNPROCESSABLE_ENTITY, message="Validation error"
            ).to_http_response()

        # Pass file and edit_model data to your service
        await CloudModelWorkflowService(session).edit_cloud_model(
            model_id=model_id, data=edit_model.dict(exclude_unset=True, exclude_none=True), file=license_file
        )

        return SuccessResponse(message="Cloud model edited successfully", code=status.HTTP_200_OK).to_http_response()
    except ClientException as e:
        logger.exception(f"Failed to edit cloud model: {e}")
        return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
    except JSONDecodeError as e:
        logger.exception(f"Failed to edit cloud model: {e}")
        return ErrorResponse(
            code=status.HTTP_422_UNPROCESSABLE_ENTITY, message="Failed to edit cloud model. Invalid input format."
        ).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to edit cloud model: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to edit cloud model"
        ).to_http_response()


# @model_router.get(
#     "/cloud-model-workflow/{workflow_id}",
#     responses={
#         status.HTTP_500_INTERNAL_SERVER_ERROR: {
#             "model": ErrorResponse,
#             "description": "Service is unavailable due to server error",
#         },
#         status.HTTP_400_BAD_REQUEST: {
#             "model": ErrorResponse,
#             "description": "Service is unavailable due to client error",
#         },
#         status.HTTP_200_OK: {
#             "model": CreateCloudModelWorkflowResponse,
#             "description": "Successfully add cloud model workflow",
#         },
#     },
#     description="Get cloud model workflow",
# )
# async def get_cloud_model_workflow(
#     current_user: Annotated[User, Depends(get_current_active_user)],
#     session: Annotated[Session, Depends(get_session)],
#     workflow_id: UUID,
# ) -> Union[CreateCloudModelWorkflowResponse, ErrorResponse]:
#     """Get cloud model workflow."""
#     try:
#         return await CloudModelWorkflowService(session).get_cloud_model_workflow(workflow_id)
#     except ClientException as e:
#         logger.exception(f"Failed to get cloud model workflow: {e}")
#         return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
#     except Exception as e:
#         logger.exception(f"Failed to get cloud model workflow: {e}")
#         return ErrorResponse(
#             code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get cloud model workflow"
#         ).to_http_response()


# @model_router.get(
#     "/cloud-models",
#     responses={
#         status.HTTP_500_INTERNAL_SERVER_ERROR: {
#             "model": ErrorResponse,
#             "description": "Service is unavailable due to server error",
#         },
#         status.HTTP_400_BAD_REQUEST: {
#             "model": ErrorResponse,
#             "description": "Service is unavailable due to client error",
#         },
#         status.HTTP_200_OK: {
#             "model": ProviderResponse,
#             "description": "Successfully list all providers",
#         },
#     },
#     description="List all cloud models",
# )
# async def list_cloud_models(
#     current_user: Annotated[User, Depends(get_current_active_user)],
#     session: Annotated[Session, Depends(get_session)],
#     filters: Annotated[CloudModelFilter, Depends()],
#     tags: List[str] = Query(default_factory=list),
#     tasks: List[str] = Query(default_factory=list),
#     page: int = Query(1, ge=1),
#     limit: int = Query(10, ge=0),
#     order_by: Optional[List[str]] = Depends(parse_ordering_fields),
#     search: bool = False,
# ) -> Union[CloudModelResponse, ErrorResponse]:
#     """List all cloud models."""
#     # Calculate offset
#     offset = (page - 1) * limit

#     # Convert UserFilter to dictionary
#     filters_dict = filters.model_dump(exclude_none=True)
#     if tags:
#         filters_dict["tags"] = tags
#     if tasks:
#         filters_dict["tasks"] = tasks

#     try:
#         db_models, count = await CloudModelService(session).get_all_cloud_models(
#             offset, limit, filters_dict, order_by, search
#         )
#     except Exception as e:
#         logger.exception(f"Failed to get all cloud models: {e}")
#         return ErrorResponse(
#             code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get all cloud models"
#         ).to_http_response()

#     return CloudModelResponse(
#         cloud_models=db_models,
#         total_record=count,
#         page=page,
#         limit=limit,
#         object="cloud_models.list",
#         code=status.HTTP_200_OK,
#     ).to_http_response()


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


@model_router.get(
    "/tags",
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
            "model": TagsListResponse,
            "description": "Successfully searched tags by name",
        },
    },
    description="Search model tags by name with pagination",
)
async def list_model_tags(
    session: Annotated[Session, Depends(get_session)],
    name: Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_active_user),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
) -> Union[TagsListResponse, ErrorResponse]:
    """List tags by name with pagination support."""
    offset = (page - 1) * limit

    try:
        db_tags, count = await ModelService(session).list_model_tags(name or "", offset, limit)
    except Exception as e:
        return ErrorResponse(code=status.HTTP_500_INTERNAL_SERVER_ERROR, message=str(e)).to_http_response()

    return TagsListResponse(
        tags=db_tags,
        total_record=count,
        page=page,
        limit=limit,
        object="tags.search",
        code=status.HTTP_200_OK,
    ).to_http_response()


@model_router.get(
    "/tasks",
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
            "model": TasksListResponse,
            "description": "Successfully listed tasks",
        },
    },
    description="Search model tags by name with pagination",
)
async def list_model_tasks(
    session: Annotated[Session, Depends(get_session)],
    name: Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_active_user),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1),
) -> Union[TasksListResponse, ErrorResponse]:
    """List tasks by name with pagination support."""
    offset = (page - 1) * limit

    try:
        db_tasks, count = await ModelService(session).list_model_tasks(name or "", offset, limit)
    except Exception as e:
        return ErrorResponse(code=status.HTTP_500_INTERNAL_SERVER_ERROR, message=str(e)).to_http_response()

    return TasksListResponse(
        tasks=db_tasks,
        total_record=count,
        page=page,
        limit=limit,
        object="tasks.list",
        code=status.HTTP_200_OK,
    ).to_http_response()


@model_router.get(
    "/authors",
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
            "model": ModelAuthorResponse,
            "description": "Successfully searched author by name",
        },
    },
    description="Search model author by name with pagination",
)
async def list_all_model_authors(
    session: Annotated[Session, Depends(get_session)],
    filters: Annotated[ModelAuthorFilter, Depends()],
    current_user: User = Depends(get_current_active_user),
    search: bool = False,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),
) -> Union[ModelAuthorResponse, ErrorResponse]:
    """Search author by name with pagination support."""
    offset = (page - 1) * limit

    filters_dict = filters.model_dump(exclude_none=True)

    try:
        db_authors, count = await ModelService(session).list_all_model_authors(
            offset, limit, filters_dict, order_by, search
        )
    except Exception as e:
        return ErrorResponse(code=status.HTTP_500_INTERNAL_SERVER_ERROR, message=str(e)).to_http_response()

    return ModelAuthorResponse(
        authors=db_authors,
        total_record=count,
        page=page,
        limit=limit,
        object="author.list",
        code=status.HTTP_200_OK,
    ).to_http_response()


@model_router.get(
    "/{model_id}",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Model not found",
        },
        status.HTTP_200_OK: {
            "model": ModelDetailSuccessResponse,
            "description": "Successfully retrieved model details",
        },
    },
    description="Retrieve details of a model by ID",
)
async def retrieve_model(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    model_id: UUID,
) -> Union[ModelDetailSuccessResponse, ErrorResponse]:
    """Retrieve details of a model by its ID."""
    try:
        return await ModelService(session).retrieve_model(model_id)
    except ClientException as e:
        logger.exception(f"Failed to get model details: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get model details: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Failed to retrieve model details",
        ).to_http_response()
