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

from typing import List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.user_ops.schemas import User

from ..commons import logging
from ..commons.async_utils import get_user_from_auth_header
from ..commons.constants import ModalityEnum
from ..commons.dependencies import (
    get_current_active_user,
    get_session,
    parse_ordering_fields,
)
from ..commons.exceptions import ClientException
from ..commons.schemas import ErrorResponse, SuccessResponse
from .schemas import (
    ChatSessionEditRequest,
    ChatSessionFilter,
    ChatSessionPaginatedResponse,
    ChatSessionSuccessResponse,
    MessageCreateRequest,
    MessageEditRequest,
    MessageFilter,
    MessagePaginatedResponse,
    MessageSuccessResponse,
    PlaygroundDeploymentFilter,
    PlaygroundDeploymentListResponse,
    ChatSettingCreate,
    ChatSettingSuccessResponse,
    ChatSettingPaginatedResponse,
    ChatSettingFilter,
    ChatSettingEditRequest,
)
from .services import ChatSessionService, MessageService, PlaygroundService, ChatSettingService


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


# @playground_router.post(
#     "/chat-sessions",
#     responses={
#         status.HTTP_200_OK: {
#             "model": ChatSessionSuccessResponse,
#             "description": "Chat session created successfully.",
#         },
#         status.HTTP_500_INTERNAL_SERVER_ERROR: {
#             "model": ErrorResponse,
#             "description": "Service is unavailable due to server error",
#         },
#         status.HTTP_400_BAD_REQUEST: {
#             "model": ErrorResponse,
#             "description": "Service is unavailable due to client error",
#         },
#     },
#     description="Create a chat session.",
# )
# async def create_chat_session(
#     request: ChatSessionCreate,
#     current_user: Annotated[User, Depends(get_current_active_user)],
#     session: Annotated[Session, Depends(get_session)],
# ) -> Union[ChatSessionSuccessResponse, ErrorResponse]:
#     """Create a new chat session."""
#     try:
#         chat_session_data = request.model_dump(exclude_unset=True)
#         db_chat_session = await ChatSessionService(session).create_chat_session(current_user.id, chat_session_data)

#     except ClientException as e:
#         logger.exception(f"Failed to create chat session: {e}")
#         return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
#     except Exception as e:
#         logger.exception(f"Failed to create chat session: {e}")
#         return ErrorResponse(
#             code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to create chat session"
#         ).to_http_response()

#     return ChatSessionSuccessResponse(
#         chat_session=db_chat_session,
#         message="chat session created successfully",
#         code=status.HTTP_200_OK,
#         object="chat_session.create",
#     ).to_http_response()


@playground_router.get(
    "/chat-sessions",
    responses={
        status.HTTP_200_OK: {
            "model": ChatSessionPaginatedResponse,
            "description": "Successfully retrieved chat sessions.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
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
        db_chat_sessions, count = await ChatSessionService(session).list_chat_sessions(
            current_user.id, offset, limit, filters_dict, order_by, search
        )
    except ClientException as e:
        logger.exception(f"Failed to list chat sessions: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to list chat sessions: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Failed to list chat sessions",
        ).to_http_response()
    return ChatSessionPaginatedResponse(
        chat_sessions=db_chat_sessions,
        total_record=count,
        page=page,
        limit=limit,
        object="chat_sessions.list",
        code=status.HTTP_200_OK,
        message="Successfully retrieved chat sessions",
    ).to_http_response()


@playground_router.get(
    "/chat-sessions/{chat_session_id}",
    responses={
        status.HTTP_200_OK: {
            "model": ChatSessionSuccessResponse,
            "description": "Successfully retrieved chat session details.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
    },
)
async def get_chat_session_details(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    chat_session_id: UUID,
) -> Union[ChatSessionSuccessResponse, ErrorResponse]:
    """Retrieve details of a specific chat session."""
    try:
        db_chat_session = await ChatSessionService(session).get_chat_session_details(chat_session_id)

    except ClientException as e:
        logger.exception(f"Failed to retrieve details of chat session: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to retrieve details of chat session: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to retrieve details of chat session"
        ).to_http_response()

    return ChatSessionSuccessResponse(
        chat_session=db_chat_session,
        message="chat session retrieved successfully",
        code=status.HTTP_200_OK,
        object="chat_session.get",
    ).to_http_response()


@playground_router.delete(
    "/chat-sessions/{chat_session_id}",
    responses={
        status.HTTP_200_OK: {
            "model": SuccessResponse,
            "description": "Chat session deleted successfully.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
    },
    description="Delete chat session",
)
async def delete_chat_session(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    chat_session_id: UUID,
) -> Union[SuccessResponse, ErrorResponse]:
    """Delete a chat session."""
    try:
        await ChatSessionService(session).delete_chat_session(chat_session_id)

    except ClientException as e:
        logger.exception(f"Failed to delete chat session: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to delete chat session: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to delete chat session"
        ).to_http_response()
    return SuccessResponse(
        code=status.HTTP_200_OK,
        message="Chat session deleted successfully",
        object="chat_session.delete",
    )


@playground_router.patch(
    "/chat-sessions/{chat_session_id}",
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
            "model": ChatSessionSuccessResponse,
            "description": "Successfully edited chat session",
        },
    },
    description="Edit chat session",
)
async def edit_chat_session(
    chat_session_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    request: ChatSessionEditRequest,
) -> Union[ChatSessionSuccessResponse, ErrorResponse]:
    """Edit chat session"""
    try:
        db_chat_session = await ChatSessionService(session).edit_chat_session(
            chat_session_id=chat_session_id, data=request.model_dump(exclude_unset=True, exclude_none=True)
        )
        return ChatSessionSuccessResponse(
            chat_session=db_chat_session,
            message="Chat session details updated successfully",
            code=status.HTTP_200_OK,
            object="chat_session.edit",
        )
    except ClientException as e:
        logger.exception(f"Failed to edit chat session: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to edit chat session: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to edit chat session"
        ).to_http_response()


@playground_router.post(
    "/messages",
    responses={
        status.HTTP_200_OK: {
            "model": MessageSuccessResponse,
            "description": "Message created successfully.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
    },
    description="Create a Message.",
)
async def create_message(
    request: MessageCreateRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
) -> Union[MessageSuccessResponse, ErrorResponse]:
    """Create a new Message."""
    try:
        message_data = request.model_dump(exclude_unset=True)
        db_message = await MessageService(session).create_message(current_user.id, message_data)

    except ClientException as e:
        logger.exception(f"Failed to create message: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to create message: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to create message"
        ).to_http_response()

    return MessageSuccessResponse(
        chat_message=db_message,
        message="message created successfully",
        code=status.HTTP_200_OK,
        object="message.create",
    ).to_http_response()


@playground_router.get(
    "/chat-sessions/{chat_session_id}/messages",
    responses={
        status.HTTP_200_OK: {
            "model": MessagePaginatedResponse,
            "description": "Successfully retrieved messages.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
    },
    description="Retrieve all messages for a given chat session.",
)
async def get_all_messages(
    chat_session_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    filters: MessageFilter = Depends(),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),
    search: bool = False,
) -> Union[MessagePaginatedResponse, ErrorResponse]:
    """Retrieve all messages for a given chat session."""
    offset = (page - 1) * limit

    filters_dict = filters.model_dump(exclude_none=True)

    try:
        db_messages, count = await MessageService(session).get_messages_by_chat_session(
            chat_session_id, filters_dict, offset, limit, order_by, search
        )
    except ClientException as e:
        logger.exception(f"Failed to retrieve messages: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to retrieve messages: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Failed to retrieve messages",
        ).to_http_response()

    return MessagePaginatedResponse(
        chat_messages=db_messages,
        total_record=count,
        page=page,
        limit=limit,
        object="messages.list",
        code=status.HTTP_200_OK,
        message="Successfully retrieved messages",
    ).to_http_response()


@playground_router.patch(
    "/messages/{message_id}",
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
            "model": MessageSuccessResponse,
            "description": "Successfully edited message",
        },
    },
    description="Edit a chat message",
)
async def edit_message(
    message_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    request: MessageEditRequest,
) -> Union[MessageSuccessResponse, ErrorResponse]:
    """Edit a chat message."""
    try:
        db_message = await MessageService(session).edit_message(
            message_id=message_id, data=request.model_dump(exclude_unset=True, exclude_none=True)
        )
        return MessageSuccessResponse(
            chat_message=db_message,
            message="Message updated successfully",
            code=status.HTTP_200_OK,
            object="message.edit",
        )
    except ClientException as e:
        logger.exception(f"Failed to edit message: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except ValidationError as e:
        logger.exception(f"ValidationErrors: {str(e)}")
        raise RequestValidationError(e.errors())
    except Exception as e:
        logger.exception(f"Failed to edit message: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to edit message"
        ).to_http_response()


@playground_router.delete(
    "/messages/{message_id}",
    responses={
        status.HTTP_200_OK: {
            "model": SuccessResponse,
            "description": "Message deleted successfully.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
    },
    description="Delete a message",
)
async def delete_message(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    message_id: UUID,
) -> Union[SuccessResponse, ErrorResponse]:
    """Delete a message."""
    try:
        await MessageService(session).delete_message(message_id)

    except ClientException as e:
        logger.exception(f"Failed to delete message: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to delete message: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to delete message"
        ).to_http_response()

    return SuccessResponse(
        code=status.HTTP_200_OK,
        message="Message deleted successfully",
        object="message.delete",
    )


@playground_router.post(
    "/chat-settings",
    responses={
        status.HTTP_200_OK: {
            "model": ChatSettingSuccessResponse,
            "description": "Chat setting created successfully.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
    },
    description="Create a chat setting.",
)
async def create_chat_setting(
    request: ChatSettingCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
) -> Union[ChatSettingSuccessResponse, ErrorResponse]:
    """Create a new chat setting."""
    try:
        chat_setting_data = request.model_dump(exclude_unset=True)
        db_chat_setting = await ChatSettingService(session).create_chat_setting(current_user.id, chat_setting_data)

    except ClientException as e:
        logger.exception(f"Failed to create chat setting: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to create chat setting: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Failed to create chat setting",
        ).to_http_response()

    return ChatSettingSuccessResponse(
        chat_setting=db_chat_setting,
        message="Chat setting created successfully",
        code=status.HTTP_200_OK,
        object="chat_setting.create",
    ).to_http_response()


@playground_router.get(
    "/chat-settings",
    responses={
        status.HTTP_200_OK: {
            "model": ChatSettingPaginatedResponse,
            "description": "Successfully retrieved chat settings.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
    },
    description="List all chat settings.",
)
async def list_chat_settings(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    filters: ChatSettingFilter = Depends(),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),
    search: bool = False,
) -> Union[ChatSettingPaginatedResponse, ErrorResponse]:
    """List all chat settings for a user."""
    offset = (page - 1) * limit

    filters_dict = filters.model_dump(exclude_none=True)

    try:
        db_chat_settings, count = await ChatSettingService(session).list_chat_settings(
            current_user.id, offset, limit, filters_dict, order_by, search
        )
    except ClientException as e:
        logger.exception(f"Failed to list chat settings: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to list chat settings: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to list chat settings"
        ).to_http_response()
    return ChatSettingPaginatedResponse(
        chat_settings=db_chat_settings,
        total_record=count,
        page=page,
        limit=limit,
        object="chat_settings.list",
        code=status.HTTP_200_OK,
        message="Successfully retrieved chat settings",
    ).to_http_response()


@playground_router.get(
    "/chat-settings/{chat_setting_id}",
    responses={
        status.HTTP_200_OK: {
            "model": ChatSettingSuccessResponse,
            "description": "Successfully retrieved chat setting details.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
    },
)
async def get_chat_setting_details(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    chat_setting_id: UUID,
) -> Union[ChatSettingSuccessResponse, ErrorResponse]:
    """Retrieve details of a specific chat setting."""
    try:
        db_chat_setting = await ChatSettingService(session).get_chat_setting_details(chat_setting_id)
    except ClientException as e:
        logger.exception(f"Failed to retrieve details of chat setting: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to retrieve details of chat setting: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to retrieve details of chat setting"
        ).to_http_response()
    return ChatSettingSuccessResponse(
        chat_setting=db_chat_setting,
        message="Chat setting retrieved successfully",
        code=status.HTTP_200_OK,
        object="chat_setting.get",
    ).to_http_response()


@playground_router.patch(
    "/chat-settings/{chat_setting_id}",
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
            "model": ChatSettingSuccessResponse,
            "description": "Successfully edited chat setting",
        },
    },
    description="Edit chat setting",
)
async def edit_chat_setting(
    chat_setting_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    request: ChatSettingEditRequest,
) -> Union[ChatSettingSuccessResponse, ErrorResponse]:
    """Edit chat setting"""
    try:
        db_chat_setting = await ChatSettingService(session).edit_chat_setting(
            chat_setting_id=chat_setting_id, data=request.model_dump(exclude_unset=True, exclude_none=True)
        )
        return ChatSettingSuccessResponse(
            chat_setting=db_chat_setting,
            message="Chat setting details updated successfully",
            code=status.HTTP_200_OK,
            object="chat_setting.edit",
        )
    except ClientException as e:
        logger.exception(f"Failed to edit chat setting: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to edit chat setting: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to edit chat setting"
        ).to_http_response()


@playground_router.delete(
    "/chat-settings/{chat_setting_id}",
    responses={
        status.HTTP_200_OK: {
            "model": SuccessResponse,
            "description": "Chat setting deleted successfully.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
    },
    description="Delete chat setting",
)
async def delete_chat_setting(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    chat_setting_id: UUID,
) -> Union[SuccessResponse, ErrorResponse]:
    """Delete a chat setting."""
    try:
        await ChatSettingService(session).delete_chat_setting(chat_setting_id)
    except ClientException as e:
        logger.exception(f"Failed to delete chat setting: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to delete chat setting: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to delete chat setting"
        ).to_http_response()
    return SuccessResponse(
        code=status.HTTP_200_OK,
        message="Chat setting deleted successfully",
        object="chat_setting.delete",
    )
