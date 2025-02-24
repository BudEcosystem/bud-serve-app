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


"""Contains core Pydantic schemas used for data validation and serialization within the playground ops services."""

from datetime import datetime

from pydantic import UUID4, BaseModel, ConfigDict, Field

from budapp.commons.schemas import PaginatedSuccessResponse, SuccessResponse


class ChatSessionCreate(BaseModel):
    # deployment_id: UUID4
    name: str = Field(default="unnamed chat")
    chat_settings_id: UUID4 | None = None
    note: list[str] | None = None


class ChatSessionResponse(BaseModel):
    id: UUID4
    user_id: UUID4
    # deployment_id: UUID4
    name: str
    chat_settings_id: UUID4 | None = None
    note: list[str] | None = None
    created_at: datetime
    modified_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ChatSessionSuccessResponse(SuccessResponse):
    session: ChatSessionResponse


class ChatSessionListResponse(BaseModel):
    id: UUID4
    name: str
    total_tokens: int | None = None  # Aggregated token count for the session
    created_at: datetime
    modified_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ChatSessionPaginatedResponse(PaginatedSuccessResponse):
    chat_sessions: list[ChatSessionListResponse] = []


class ChatSessionFilter(BaseModel):
    # user_id: UUID4  #check if req
    name: str | None = None
