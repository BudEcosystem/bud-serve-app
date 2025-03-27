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

"""The endpoint ops package, containing essential business logic, services, and routing configurations for the endpoint ops."""

from typing import List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons import logging
from budapp.commons.dependencies import (
    get_current_active_user,
    get_session,
    parse_ordering_fields,
)
from budapp.commons.exceptions import ClientException
from budapp.user_ops.schemas import User

from ..commons.schemas import ErrorResponse, SuccessResponse
from ..workflow_ops.schemas import RetrieveWorkflowDataResponse
from ..workflow_ops.services import WorkflowService
from .schemas import (
    AddWorkerRequest,
    DeleteWorkerRequest,
    EndpointFilter,
    EndpointPaginatedResponse,
    ModelClusterDetailResponse,
    WorkerDetailResponse,
    WorkerInfoFilter,
    WorkerInfoResponse,
    WorkerLogsResponse,
    WorkerMetricsResponse,
)
from .services import EndpointService


logger = logging.get_logger(__name__)

endpoint_router = APIRouter(prefix="/endpoints", tags=["endpoint"])


@endpoint_router.get(
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
            "model": EndpointPaginatedResponse,
            "description": "Successfully list all endpoints",
        },
    },
    description="List all endpoints. \n\n order_by fields are: name, status, created_at, modified_at, cluster_name, model_name, modality",
)
async def list_all_endpoints(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    filters: Annotated[EndpointFilter, Depends()],
    project_id: UUID = Query(description="List endpoints by project id"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),
    search: bool = False,
) -> Union[EndpointPaginatedResponse, ErrorResponse]:
    """List all endpoints."""
    # Calculate offset
    offset = (page - 1) * limit

    # Construct filters
    filters_dict = filters.model_dump(exclude_none=True, exclude_unset=True)

    try:
        db_endpoints, count = await EndpointService(session).get_all_endpoints(
            project_id, offset, limit, filters_dict, order_by, search
        )
    except ClientException as e:
        logger.exception(f"Failed to get all endpoints: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get all endpoints: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get all endpoints"
        ).to_http_response()

    return EndpointPaginatedResponse(
        endpoints=db_endpoints,
        total_record=count,
        page=page,
        limit=limit,
        object="endpoints.list",
        code=status.HTTP_200_OK,
        message="Successfully list all endpoints",
    ).to_http_response()


@endpoint_router.post(
    "/{endpoint_id}/delete-workflow",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Invalid request parameters",
        },
        status.HTTP_200_OK: {
            "model": SuccessResponse,
            "description": "Successfully executed delete endpoint workflow",
        },
    },
    description="Delete an endpoint by ID",
)
async def delete_endpoint(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    endpoint_id: UUID,
) -> Union[SuccessResponse, ErrorResponse]:
    """Delete a endpoint by its ID."""
    try:
        db_workflow = await EndpointService(session).delete_endpoint(endpoint_id, current_user.id)
        logger.debug(f"Endpoint deleting initiated with workflow id: {db_workflow.id}")
        return SuccessResponse(
            message="Deployment deleting initiated successfully",
            code=status.HTTP_200_OK,
            object="endpoint.delete",
        )
    except ClientException as e:
        logger.exception(f"Failed to delete endpoint: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to delete endpoint: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Failed to delete endpoint",
        ).to_http_response()


@endpoint_router.get(
    "/{endpoint_id}/workers",
    responses={
        status.HTTP_200_OK: {
            "model": WorkerInfoResponse,
            "description": "Successfully get endpoint detail",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Failed to get endpoint workers",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Endpoint not found",
        },
    },
)
async def get_endpoint_workers(
    endpoint_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    filters: Annotated[WorkerInfoFilter, Depends()],
    refresh: bool = Query(False),  # noqa: B008
    page: int = Query(1, ge=1),  # noqa: B008
    limit: int = Query(10, ge=0),  # noqa: B008
    order_by: Optional[List[str]] = Query(None),  # noqa: B008
    search: bool = Query(False),  # noqa: B008
) -> Union[WorkerInfoResponse, ErrorResponse]:
    """Get endpoint workers."""
    try:
        workers = await EndpointService(session).get_endpoint_workers(
            endpoint_id, filters, refresh, page, limit, order_by, search
        )
        response = WorkerInfoResponse(**workers)
    except ClientException as e:
        logger.exception(f"Failed to get endpoint workers: {e}")
        response = ErrorResponse(message=e.message, code=e.status_code)
    except Exception as e:
        logger.exception(f"Failed to get endpoint workers: {e}")
        response = ErrorResponse(message="Failed to get endpoint workers", code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return response.to_http_response()


@endpoint_router.get(
    "/{endpoint_id}/workers/{worker_id}/logs",
    responses={
        status.HTTP_200_OK: {
            "model": WorkerLogsResponse,
            "description": "Successfully get endpoint detail",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Failed to get endpoint workers",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Worker not found",
        },
    },
)
async def get_endpoint_worker_logs(
    endpoint_id: UUID,
    worker_id: UUID,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> Union[WorkerLogsResponse, ErrorResponse]:
    """Get endpoint worker logs."""
    try:
        worker_log = await EndpointService(session).get_endpoint_worker_logs(endpoint_id, worker_id)
        response = WorkerLogsResponse(
            logs=worker_log,
            object="endpoint.worker.logs",
            message="Successfully fetched endpoint worker logs",
            code=status.HTTP_200_OK,
        )
    except ClientException as e:
        logger.exception(f"Failed to get endpoint worker detail: {e}")
        response = ErrorResponse(message=e.message, code=e.status_code)
    except Exception as e:
        logger.exception(f"Failed to get endpoint worker detail: {e}")
        response = ErrorResponse(
            message="Failed to get endpoint worker detail", code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    return response.to_http_response()


@endpoint_router.get(
    "/{endpoint_id}/workers/{worker_id}/metrics",
    responses={
        status.HTTP_200_OK: {
            "model": WorkerMetricsResponse,  # noqa: F821
            "description": "Successfully get endpoint worker metrics",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Worker not found",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Failed to get endpoint worker metrics",
        },
    },
)
async def get_endpoint_worker_metrics(
    endpoint_id: UUID,
    worker_id: UUID,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> Union[WorkerMetricsResponse, ErrorResponse]:
    """Get endpoint worker metrics."""
    try:
        worker_metrics = await EndpointService(session).get_worker_metrics_history(endpoint_id, worker_id)
        response = WorkerMetricsResponse(
            metrics=worker_metrics,
            object="endpoint.worker.metrics",
            message="Successfully fetched endpoint worker metrics",
            code=status.HTTP_200_OK,
        )
    except ClientException as e:
        logger.exception(f"Failed to get endpoint worker metrics: {e}")
        response = ErrorResponse(message=e.message, code=e.status_code)
    except Exception as e:
        logger.exception(f"Failed to get endpoint worker metrics: {e}")
        response = ErrorResponse(message="Failed to get endpoint worker metrics", code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return response.to_http_response()


@endpoint_router.get(
    "/{endpoint_id}/workers/{worker_id}",
    responses={
        status.HTTP_200_OK: {
            "model": WorkerDetailResponse,
            "description": "Successfully get endpoint detail",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Failed to get endpoint workers",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Worker not found",
        },
    },
)
async def get_endpoint_worker_detail(
    endpoint_id: UUID,
    worker_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    reload: bool = Query(False),  # noqa: B008
) -> Union[WorkerDetailResponse, ErrorResponse]:
    """Get endpoint workers."""
    try:
        worker_detail = await EndpointService(session).get_endpoint_worker_detail(endpoint_id, worker_id, reload)
        response = WorkerDetailResponse(**worker_detail)
    except ClientException as e:
        logger.exception(f"Failed to get endpoint worker detail: {e}")
        response = ErrorResponse(message=e.message, code=e.status_code)
    except Exception as e:
        logger.exception(f"Failed to get endpoint worker detail: {e}")
        response = ErrorResponse(
            message="Failed to get endpoint worker detail", code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    return response.to_http_response()


@endpoint_router.get(
    "/{endpoint_id}/model-cluster-detail",
    responses={
        status.HTTP_200_OK: {
            "model": ModelClusterDetailResponse,
            "description": "Successfully get model cluster detail",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Endpoint not found",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Failed to get model cluster detail",
        },
    },
)
async def get_model_cluster_detail(
    endpoint_id: UUID,
    _: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
) -> Union[ModelClusterDetailResponse, ErrorResponse]:
    """Get model cluster detail."""
    try:
        model_cluster_detail = await EndpointService(session).get_model_cluster_detail(endpoint_id)
        response = ModelClusterDetailResponse(
            object="endpoint.detail",
            result=model_cluster_detail,
            message="Successfully fetched model cluster detail for the deployment.",
        )
    except ClientException as e:
        logger.exception(f"Failed to get model cluster detail: {e}")
        response = ErrorResponse(message=e.message, code=e.status_code)
    except Exception as e:
        logger.exception(f"Failed to get model cluster detail: {e}")
        response = ErrorResponse(
            message="Failed to get model cluster detail", code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    return response.to_http_response()


@endpoint_router.post(
    "/delete-worker",
    responses={
        status.HTTP_200_OK: {
            "model": SuccessResponse,
            "description": "Successfully deleted deploymentworker",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Worker not found",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Failed to delete deployment worker",
        },
    },
)
async def delete_endpoint_worker(
    request: DeleteWorkerRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
) -> Union[SuccessResponse, ErrorResponse]:
    """Delete a endpoint worker by its ID."""
    try:
        db_workflow = await EndpointService(session).delete_endpoint_worker(request.endpoint_id, request.worker_id, request.worker_name, current_user.id)
        logger.debug(f"Endpoint deleting initiated with workflow id: {db_workflow.id}")
        response = SuccessResponse(
            message="Worker deleting initiated successfully",
            code=status.HTTP_200_OK,
            object="worker.delete",
        )
    except ClientException as e:
        logger.exception(f"Failed to get endpoint worker detail: {e}")
        response = ErrorResponse(message=e.message, code=e.status_code)
    except Exception as e:
        logger.exception(f"Failed to get endpoint worker detail: {e}")
        response = ErrorResponse(message="Failed to get endpoint worker detail", code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return response.to_http_response()


@endpoint_router.post(
    "/add-worker",
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
            "description": "Successfully add worker",
        },
    },
    description="Add worker to endpoint",
)
async def add_worker_to_endpoint(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    request: AddWorkerRequest,
) -> Union[RetrieveWorkflowDataResponse, ErrorResponse]:
    """Add worker to endpoint."""
    try:
        db_workflow = await EndpointService(session).add_worker_to_endpoint_workflow(
            current_user_id=current_user.id,
            request=request,
        )

        return await WorkflowService(session).retrieve_workflow_data(db_workflow.id)
    except ClientException as e:
        logger.exception(f"Failed to add worker to endpoint: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to add worker to endpoint: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to add worker to endpoint"
        ).to_http_response()

