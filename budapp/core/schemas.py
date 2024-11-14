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


"""Contains core Pydantic schemas used for data validation and serialization within the core services."""

from typing import Any, Dict, List, Optional, Self, Union

from pydantic import BaseModel, model_validator, field_validator

from budapp.commons import logging
from budapp.commons.constants import NotificationCategory, NotificationType, ModelTemplateTypeEnum
from budapp.commons.schemas import CloudEventBase, SuccessResponse


logger = logging.get_logger(__name__)


class IconBase(BaseModel):
    """Base icon schema."""

    name: str
    file_path: str
    category: str


class IconCreate(IconBase):
    """Create icon schema."""

    pass


class IconUpdate(BaseModel):
    """Update icon schema."""

    category: str
    name: str


# Schemas related to notifications


class NotificationContent(BaseModel):
    """Represents the content of a notification."""

    title: str | None = None
    message: str | None = None
    status: str | None = None
    result: Optional[Dict[str, Any]] = None
    primary_action: str | None = None
    secondary_action: str | None = None


class NotificationPayload(BaseModel):
    """Schema for notification payload."""

    category: NotificationCategory
    type: str | None = None
    event: str | None = None
    workflow_id: str | None = None
    source: str
    content: NotificationContent


class NotificationRequest(CloudEventBase):
    """Represents a notification request."""

    notification_type: NotificationType = NotificationType.EVENT
    name: str  # Workflow identifier
    subscriber_ids: Optional[Union[str, List[str]]] = None
    actor: Optional[str] = None
    topic_keys: Optional[Union[str, List[str]]] = None
    payload: NotificationPayload

    @model_validator(mode="before")
    def log_notification_hits(cls, data):
        """Log the notification hits for debugging purposes."""
        # TODO: remove this function after Debugging
        logger.info("================================================")
        logger.info("Received hit in notifications/:")
        logger.info(f"{data}")
        logger.info("================================================")
        return data

    @model_validator(mode="after")
    def validate_notification_rules(self) -> Self:
        """Check if required fields are present in the request.

        Raises:
            ValueError: If `subscriber_ids` is not present for event notifications.
            ValueError: If `topic_keys` is not present for topic notifications.

        Returns:
            Self: The instance of the class.
        """
        if self.notification_type == NotificationType.EVENT and not self.subscriber_ids:
            raise ValueError("subscriber_ids is required for event notifications")
        if self.notification_type == NotificationType.TOPIC and not self.topic_keys:
            raise ValueError("topic_keys is required for topic notifications")
        if self.notification_type == NotificationType.BROADCAST and (self.subscriber_ids or self.topic_keys):
            raise ValueError("subscriber_ids and topic_keys are not allowed for broadcast notifications")

        return self


class NotificationResponse(SuccessResponse):
    """Represents a notification response."""

    pass

class ModelTemplateCreate(BaseModel):
    """Model template create schema"""

    name: str
    description: str
    icon: str
    template_type: ModelTemplateTypeEnum
    avg_sequence_length: Optional[int] = None
    avg_context_length: Optional[int] = None
    per_session_tokens_per_sec: Optional[list[int]] = None
    ttft: Optional[list[int]] = None
    e2e_latency: Optional[list[int]] = None

    @field_validator("per_session_tokens_per_sec", "ttft", "e2e_latency", mode="before")
    @classmethod
    def validate_int_range(cls, value):
        if value is not None and (
            not isinstance(value, list)
            or len(value) != 2
            or not all(isinstance(x, int) for x in value)
        ):
            raise ValueError("Must be a list of two integers")
        return value

class ModelTemplateUpdate(ModelTemplateCreate):
    """Model template update schema"""

    pass
