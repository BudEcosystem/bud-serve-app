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

"""The playground ops services. Contains business logic for playground ops."""

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import status

from ..commons import logging
from ..commons.db_utils import SessionMixin
from ..commons.exceptions import ClientException
from ..commons.constants import EndpointStatusEnum
from ..credential_ops.crud import CredentialDataManager
from ..credential_ops.models import Credential as CredentialModel
from ..endpoint_ops.crud import EndpointDataManager
from ..endpoint_ops.models import Endpoint as EndpointModel
from ..project_ops.crud import ProjectDataManager
from .crud import ChatSessionDataManager, MessageDataManager
from .models import ChatSession, Message
from .schemas import ChatSessionCreate, ChatSessionListResponse, MessageResponse


logger = logging.get_logger(__name__)


class PlaygroundService(SessionMixin):
    """Playground service."""

    async def get_all_playground_deployments(
        self,
        current_user_id: Optional[UUID] = None,
        api_key: Optional[str] = None,
        offset: int = 0,
        limit: int = 10,
        filters: Optional[Dict] = None,
        order_by: Optional[List] = None,
        search: bool = False,
    ) -> Tuple[List[EndpointModel], int]:
        """Get all playground deployments."""
        filters = filters or {}
        order_by = order_by or []

        project_ids = await self._get_authorized_project_ids(current_user_id, api_key)
        logger.debug("authorized project_ids: %s", project_ids)

        db_endpoints, count = await EndpointDataManager(self.session).get_all_playground_deployments(
            project_ids,
            offset,
            limit,
            filters,
            order_by,
            search,
        )
        logger.debug("found %s deployments", count)

        return db_endpoints, count

    async def _get_authorized_project_ids(
        self, current_user_id: Optional[UUID] = None, api_key: Optional[str] = None
    ) -> List[UUID]:
        """Get all authorized project ids."""
        if current_user_id:
            # NOTE: As per user permissions list the playground deployments (accessible project ids)
            # TODO: Query all accessible project ids for the user (Currently all active project ids since permissions are not implemented)
            logger.debug(f"Getting all playground deployments for user {current_user_id}")
            return await ProjectDataManager(self.session).get_all_active_project_ids()
        elif api_key:
            # if api_key is present identify the project id
            db_credential = await CredentialDataManager(self.session).retrieve_by_fields(
                CredentialModel, fields={"key": api_key}, missing_ok=True
            )

            if not db_credential:
                logger.error("Invalid API key found")
                raise ClientException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    message="Invalid API key",
                )
            else:
                return [db_credential.project.id]
        else:
            raise ClientException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                message="Unauthorized to access this resource",
            )


class ChatSessionService(SessionMixin):
    """Chat Session Service"""

    async def create_chat_session(self, user_id: UUID, chat_session_data: dict) -> ChatSession:
        """Create a new chat session and insert it into the database."""
        chat_session_data["user_id"] = user_id

        chat_session = ChatSession(**chat_session_data)

        db_chat_session = await ChatSessionDataManager(self.session).insert_one(chat_session)

        return db_chat_session

    async def list_chat_sessions(
        self,
        user_id: UUID,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[ChatSessionListResponse], int]:
        """List all chat sessions for a given user."""
        db_results, count = await ChatSessionDataManager(self.session).get_all_chat_sessions(
            user_id, offset, limit, filters, order_by, search
        )
        chat_sessions = []
        for db_result in db_results:
            db_chat_session = db_result[0]
            chat_session = ChatSessionListResponse(
                id=db_chat_session.id,
                name=db_chat_session.name,
                total_tokens=db_result[1],
                created_at=db_chat_session.created_at,
                modified_at=db_chat_session.modified_at,
            )
            chat_sessions.append(chat_session)
        return chat_sessions, count

    async def get_chat_session_details(self, chat_session_id: UUID) -> ChatSession:
        """Retrieve details of a session by its ID."""
        db_chat_session = await ChatSessionDataManager(self.session).retrieve_by_fields(
            ChatSession,
            fields={"id": chat_session_id},
        )

        return db_chat_session

    async def delete_chat_session(self, chat_session_id: UUID) -> None:
        """Delete chat session."""
        db_chat_session = await ChatSessionDataManager(self.session).retrieve_by_fields(
            ChatSession,
            fields={"id": chat_session_id},
        )

        await ChatSessionDataManager(self.session).delete_one(db_chat_session)

        return

    async def edit_chat_session(self, chat_session_id: UUID, data: Dict[str, Any]) -> ChatSession:
        """Edit chat session by validating and updating specific fields."""
        # Retrieve existing chat session
        db_chat_session = await ChatSessionDataManager(self.session).retrieve_by_fields(
            ChatSession,
            fields={"id": chat_session_id},
        )

        db_chat_session = await ChatSessionDataManager(self.session).update_by_fields(db_chat_session, data)

        return db_chat_session


class MessageService(SessionMixin):
    """Message Service"""

    async def create_message(self, user_id: UUID, message_data: dict) -> Message:
        """Create a new message and insert it into the database."""

        # validate deployment id
        # db_endpoint = await EndpointDataManager(self.session).retrieve_by_fields(
        #     EndpointModel,
        #     fields={"id": message_data["deployment_id"]},
        #     exclude_fields={"status": EndpointStatusEnum.DELETED},
        # )
        # If chat_session_id is not provided, create a new chat session first
        if not message_data.get("chat_session_id"):
            chat_session_data = ChatSessionCreate(name=None).model_dump(exclude_unset=True)
            chat_session_data["user_id"] = user_id
            chat_session = ChatSession(**chat_session_data)
            db_chat_session = await ChatSessionDataManager(self.session).insert_one(chat_session)
            message_data["chat_session_id"] = db_chat_session.id  # Assign the new session ID
            message_data["parent_message_id"] = None
        else:
            # validate chat session id
            db_chat_session = await ChatSessionDataManager(self.session).retrieve_by_fields(
                ChatSession, fields={"id": message_data["chat_session_id"]}
            )
            # Fetch the last message in the session to determine parent_id
            last_db_message = await MessageDataManager(self.session).get_last_message(message_data["chat_session_id"])
            message_data["parent_message_id"] = last_db_message.id if last_db_message else None

        # Create a new message
        message = Message(**message_data)
        db_message = await MessageDataManager(self.session).insert_one(message)

        return db_message

    async def get_messages_by_chat_session(
        self,
        chat_session_id: UUID,
        filters: Dict,
        offset: int = 0,
        limit: int = 10,
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[MessageResponse], int]:
        """Retrieve messages based on provided filters."""
        db_chat_session = await ChatSessionDataManager(self.session).retrieve_by_fields(
            ChatSession, fields={"id": chat_session_id}
        )

        db_messages, count = await MessageDataManager(self.session).get_messages(
            chat_session_id, filters, offset, limit, order_by, search
        )

        return db_messages, count

    async def edit_message(self, message_id: UUID, data: Dict[str, Any]) -> Message:
        """Edit a message by validating and updating specific fields."""
        # Retrieve existing message
        db_message = await MessageDataManager(self.session).retrieve_by_fields(
            Message,
            fields={"id": message_id},
        )
        if data.get("prompt"):
            # Retrieve the child message if it exists
            child_message = await MessageDataManager(self.session).retrieve_by_fields(
                Message, fields={"parent_message_id": message_id}, missing_ok=True
            )

            # Delete the child message if it exists
            if child_message:
                await MessageDataManager(self.session).delete_one(child_message)

        # Update the message with new data
        db_message = await MessageDataManager(self.session).update_by_fields(db_message, data)

        return db_message

    async def delete_message(self, message_id: UUID) -> None:
        """Delete a message and its child messages."""
        # Retrieve the message by ID
        db_message = await MessageDataManager(self.session).retrieve_by_fields(
            Message,
            fields={"id": message_id},
        )

        # Delete the message
        await MessageDataManager(self.session).delete_one(db_message)

        return
