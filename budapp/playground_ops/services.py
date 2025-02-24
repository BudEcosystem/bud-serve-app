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

from typing import Dict, List, Tuple
from uuid import UUID

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin

from .crud import ChatSessionDataManager
from .models import ChatSession
from .schemas import ChatSessionListResponse, ChatSessionResponse


logger = logging.get_logger(__name__)


class ChatSessionService(SessionMixin):
    """Chat Session Service"""

    async def create_chat_session(self, data: dict) -> ChatSession:
        """Create a new chat session and insert it into the database."""
        # is it req to check for duplicate name?

        await ChatSessionDataManager(self.session).validate_fields(ChatSession, data)

        chat_session = ChatSession(**data)

        db_chat_session = await ChatSessionDataManager(self.session).insert_one(chat_session)
        chat_session_response = ChatSessionResponse.model_validate(db_chat_session)

        return chat_session_response

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
        results, count = await ChatSessionDataManager(self.session).get_all_sessions(
            user_id, offset, limit, filters, order_by, search
        )
        updated_sessions = []
        for session_obj in results:
            updated_session = ChatSessionListResponse(
                id=session_obj.id,
                name=session_obj.name,
                total_tokens=10,  # dummy
                created_at=session_obj.created_at,
                modified_at=session_obj.modified_at,
            )
            updated_sessions.append(updated_session)
        return updated_sessions, count

    async def get_chat_session_details(self, chat_session_id: UUID) -> ChatSessionResponse:
        """Retrieve details of a session by its ID."""
        db_chat_session = await ChatSessionDataManager(self.session).retrieve_by_fields(
            ChatSession,
            fields={"id": chat_session_id},
        )
        chat_session_response = ChatSessionResponse.model_validate(db_chat_session)

        return chat_session_response

    async def delete_chat_session(self, chat_session_id: UUID) -> None:
        """Delete chat session."""
        db_chat_session = await ChatSessionDataManager(self.session).retrieve_by_fields(
            ChatSession,
            fields={"id": chat_session_id},
        )

        await ChatSessionDataManager(self.session).delete_one(db_chat_session)

        return
