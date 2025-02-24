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

"""The playground ops package, containing essential business logic, services, and routing configurations for the playground ops."""

from typing import List, Union, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, status
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from ..commons import logging
from ..commons.async_utils import get_user_from_auth_header
from ..commons.constants import ModalityEnum
from ..commons.dependencies import (
    get_current_active_user,
    get_session,
    parse_ordering_fields,
)
from ..commons.exceptions import ClientException
from budapp.user_ops.schemas import User
from ..commons.schemas import ErrorResponse, SuccessResponse
from .schemas import (
    PlaygroundDeploymentFilter,
    PlaygroundDeploymentListResponse,
    ChatSessionCreate,
    ChatSessionFilter,
    ChatSessionPaginatedResponse,
    ChatSessionResponse,
    ChatSessionSuccessResponse,
)
from .services import PlaygroundService, ChatSessionService


logger = logging.get_logger(__name__)

playground_router = APIRouter(prefix="/playground", tags=["playground"])


@playground_router.get(
    "/deployments",
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
            "model": PlaygroundDeploymentListResponse,
            "description": "Successfully list all playground deployments",
        },
    },
    description="List all playground deployments with filtering and pagination.",
)
async def list_playground_deployments(
    session: Annotated[Session, Depends(get_session)],
    filters: Annotated[PlaygroundDeploymentFilter, Depends()],
    modality: List[ModalityEnum] = Query(default=[]),
    tags: List[str] = Query(default=[]),
    tasks: List[str] = Query(default=[]),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    authorization: Annotated[
        str | None, Header()
    ] = None,  # NOTE: Can't use in Openapi docs https://github.com/fastapi/fastapi/issues/612#issuecomment-547886504
    api_key: Annotated[str | None, Header()] = None,
    order_by: Annotated[List[str] | None, Depends(parse_ordering_fields)] = None,
    search: bool = False,
) -> Union[PlaygroundDeploymentListResponse, ErrorResponse]:
    """List all playground deployments with filtering and pagination."""
    try:
        if not authorization and not api_key:
            raise ClientException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                message="Unauthorized to access this resource",
            )

        # Get current user id, if authorization header is provided
        current_user_id = None
        if authorization:
            current_user = await get_user_from_auth_header(authorization, session)
            current_user_id = current_user.id

        # Calculate offset
        offset = (page - 1) * limit

        # Convert PlaygroundDeploymentFilter to dictionary
        filters_dict = filters.model_dump(exclude_none=True)

        # Update filters_dict only for non-empty lists
        filter_updates = {"tags": tags, "tasks": tasks, "modality": modality}
        filters_dict.update({k: v for k, v in filter_updates.items() if v})

        # Get all playground deployments
        db_endpoints, count = await PlaygroundService(session).get_all_playground_deployments(
            current_user_id,
            api_key,
            offset,
            limit,
            filters_dict,
            order_by,
            search,
        )
        return PlaygroundDeploymentListResponse(
            endpoints=db_endpoints,
            total_record=count,
            page=page,
            limit=limit,
            object="playground.deployments.list",
            code=status.HTTP_200_OK,
        ).to_http_response()
    except ClientException as e:
        logger.exception(f"Failed to get all playground deployments: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get all playground deployments: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get all playground deployments"
        ).to_http_response()


@playground_router.post(
    "/session",
    responses={
        status.HTTP_200_OK: {
            "model": ChatSessionResponse,
            "description": "Chat session created successfully.",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Invalid request parameters.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error.",
        },
    },
    description="Create a chat session.",
)
async def create_chat_session(
    request: ChatSessionCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
) -> Union[ChatSessionSuccessResponse, ErrorResponse]:
    """Create a new chat session."""
    try:
        data = request.model_dump(exclude_unset=True)
        data["user_id"] = current_user.id
        db_chat_session = await ChatSessionService(session).create_chat_session(data)

    except ClientException as e:
        logger.exception(f"Failed to create chat session: {e}")
        return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to create chat session: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to create chat session"
        ).to_http_response()

    return ChatSessionSuccessResponse(
        session=db_chat_session,
        message="chat session created successfully",
        code=status.HTTP_200_OK,
        object="session.create",
    ).to_http_response()


@playground_router.get(
    "/",
    responses={
        status.HTTP_200_OK: {
            "model": ChatSessionPaginatedResponse,
            "description": "Successfully retrieved chat sessions.",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Invalid request parameters.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error.",
        },
    },
    description="List chat sessions for a user filtered by user ID.",
)
async def list_chat_sessions(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    filters: ChatSessionFilter = Depends(),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),
    search: bool = False,
) -> Union[ChatSessionPaginatedResponse, ErrorResponse]:
    """List chat sessions for a user filtered by user ID."""
    offset = (page - 1) * limit

    filters_dict = filters.model_dump(exclude_none=True)

    try:
        db_sessions, count = await ChatSessionService(session).list_chat_sessions(
            current_user.id, offset, limit, filters_dict, order_by, search
        )
    except ClientException as e:
        logger.exception(f"Failed to list chat sessions: {e}")
        return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to list chat sessions: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Failed to list chat sessions",
        ).to_http_response()
    return ChatSessionPaginatedResponse(
        chat_sessions=db_sessions,
        total_record=count,
        page=page,
        limit=limit,
        object="chat_sessions.list",
        code=status.HTTP_200_OK,
        message="Successfully retrieved chat sessions",
    ).to_http_response()


@playground_router.get(
    "/{session_id}",
    responses={
        status.HTTP_200_OK: {
            "model": ChatSessionResponse,
            "description": "Successfully retrieved chat session details.",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Chat session not found.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Internal Server Error.",
        },
    },
)
async def get_chat_session_details(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    session_id: UUID,
) -> Union[ChatSessionResponse, ErrorResponse]:
    """Retrieve details of a specific chat session."""
    try:
        db_chat_session = await ChatSessionService(session).get_chat_session_details(session_id)

    except ClientException as e:
        logger.exception(f"Failed to retrieve details of chat session: {e}")
        return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to retrieve details of chat session: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to retrieve details of chat session"
        ).to_http_response()

    return ChatSessionSuccessResponse(
        session=db_chat_session,
        message="chat session retrieved successfully",
        code=status.HTTP_200_OK,
        object="session.retrieve",
    ).to_http_response()


@playground_router.delete(
    "/{session_id}",
    responses={
        status.HTTP_200_OK: {
            "model": SuccessResponse,
            "description": "Chat session deleted successfully.",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Chat session not found.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Internal Server Error.",
        },
    },
    description="Delete chat session",
)
async def delete_chat_session(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    session_id: UUID,
) -> Union[SuccessResponse, ErrorResponse]:
    """Delete a chat session."""
    try:
        await ChatSessionService(session).delete_chat_session(session_id)

    except ClientException as e:
        logger.exception(f"Failed to delete chat session: {e}")
        return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to delete chat session: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to delete chat session"
        ).to_http_response()
    return SuccessResponse(
        code=status.HTTP_200_OK,
        message="Chat session deleted successfully",
        object="session.delete",
    )
