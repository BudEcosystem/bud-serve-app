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


"""Contains core Pydantic schemas used for data validation and serialization within the model ops services."""

import re
from datetime import datetime
from typing import Any, Optional

from pydantic import UUID4, BaseModel, ConfigDict, field_validator, model_validator

from ..commons.constants import EndpointStatusEnum
from ..commons.schemas import PaginatedSuccessResponse, SuccessResponse
from ..model_ops.schemas import ModelResponse
from ..project_ops.schemas import ProjectResponse
from ..commons.constants import FeedbackEnum


class EndpointListResponse(BaseModel):
    """Endpoint list response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID4
    name: str
    status: EndpointStatusEnum
    model: ModelResponse
    project: ProjectResponse
    created_at: datetime
    modified_at: datetime


class PlaygroundDeploymentListResponse(PaginatedSuccessResponse):
    """Playground deployment list response schema."""

    endpoints: list[EndpointListResponse] = []


class PlaygroundDeploymentFilter(BaseModel):
    """Playground deployment filter schema."""

    model_config = ConfigDict(protected_namespaces=())

    name: str | None = None
    status: EndpointStatusEnum | None = None
    model_name: str | None = None
    model_size: str | None = None

    @field_validator("model_size")
    def parse_model_size(cls, v: Optional[str]) -> Optional[int]:
        """Convert the model size string to a number in billions."""
        if not v:
            return None

        try:
            # Match only if string starts with digits and contains only digits
            match = re.match(r"^\d+$", v.strip())
            if match:
                return int(match.group())
            return None
        except Exception:
            return None


class ChatSessionCreate(BaseModel):
    """Chat session create schema"""

    name: str | None = None

    @field_validator("name", mode="before")
    @classmethod
    def set_default_name(cls, value: str | None) -> str:
        return value or "Unnamed Chat"


class ChatSessionResponse(BaseModel):
    """Chat session response schema"""

    id: UUID4
    name: str
    chat_setting: Any | None = None  # update to ChatSettingResponse when relationship is created with chat setting
    # note: list[str] | None = None
    created_at: datetime
    modified_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ChatSessionSuccessResponse(SuccessResponse):
    """Chat session success response schema"""

    chat_session: ChatSessionResponse


class ChatSessionListResponse(BaseModel):
    """Chat session list response schema"""

    id: UUID4
    name: str
    total_tokens: int  # not optional
    created_at: datetime
    modified_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class ChatSessionPaginatedResponse(PaginatedSuccessResponse):
    """Chat session paginated response schema"""

    chat_sessions: list[ChatSessionListResponse] = []


class ChatSessionFilter(BaseModel):
    """Chat session filter schema"""

    name: str | None = None


class ChatSessionEditRequest(BaseModel):
    """Chat session edit schema"""

    name: str | None = None
    chat_setting_id: UUID4 | None = None


class MessageBase(BaseModel):
    """Base schema for Message model containing shared attributes"""

    prompt: str
    response: list[dict]
    deployment_id: UUID4

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    token_per_sec: float | None = None
    ttft: float | None = None
    tpot: float | None = None
    e2e_latency: float | None = None
    is_cache: bool = False
    # harmfullness: float | None = None
    # faithfulness: float | None = None


class MessageCreateRequest(MessageBase):
    """Schema for creating a message"""

    chat_session_id: UUID4 | None = None


class MessageResponse(MessageBase):
    """Schema for returning a message response"""

    id: UUID4
    chat_session_id: UUID4
    # parent_message_id: UUID4 | None = None
    harmfullness: float | None = None
    faithfulness: float | None = None
    feedback: FeedbackEnum | None = None
    created_at: datetime
    modified_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class MessageSuccessResponse(SuccessResponse):
    """Chat session success response schema"""

    chat_message: MessageResponse


class MessagePaginatedResponse(PaginatedSuccessResponse):
    """Paginated response schema for retrieving messages"""

    chat_messages: list[MessageResponse] = []


class MessageFilter(BaseModel):
    prompt: str | None = None


class MessageEditRequest(BaseModel):
    """Message edit schema."""

    prompt: str | None = None
    response: list[dict] | None = None
    deployment_id: UUID4 | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    token_per_sec: float | None = None
    ttft: float | None = None
    tpot: float | None = None
    e2e_latency: float | None = None
    is_cache: bool | None = None
    harmfullness: float | None = None
    faithfulness: float | None = None
    feedback: FeedbackEnum | None = None

    @model_validator(mode="after")
    def validate_edit_mode(cls, values):
        """Ensure required fields are set correctly based on the type of edit."""

        prompt = values.prompt
        response = values.response
        is_content_edit = prompt is not None or response is not None
        is_feedback_edit = values.feedback is not None

        # Content edit: Both prompt & response must be provided together
        if is_content_edit:
            if prompt is None or response is None:
                raise ValueError("Both 'prompt' and 'response' must be provided for content edits.")

            required_fields = [
                "deployment_id",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "token_per_sec",
                "ttft",
                "tpot",
                "e2e_latency",
                "is_cache",
            ]
            missing_fields = [field for field in required_fields if getattr(values, field, None) is None]

            if missing_fields:
                raise ValueError(f"Missing required fields for content edit: {missing_fields}")

        # Ensure at least one field is updated
        if not is_content_edit and not is_feedback_edit:
            raise ValueError("At least one of fields prompt and response, or feedback must be provided for update.")

        return values
