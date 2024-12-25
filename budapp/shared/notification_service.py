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

"""Provides shared functions for managing notification service."""

from typing import List, Optional, Union

import aiohttp

from ..commons import logging
from ..commons.config import app_settings
from ..commons.constants import BUD_NOTIFICATION_WORKFLOW, NotificationCategory, NotificationStatus
from ..core.schemas import NotificationContent, NotificationPayload, NotificationRequest


logger = logging.get_logger(__name__)


class NotificationBuilder:
    """Builder class for notification."""

    def __init__(self):
        """Initialize the builder."""
        self.content = None
        self.payload = None

    def set_content(
        self,
        *,
        title: Optional[str] = None,
        message: Optional[str] = None,
        icon: Optional[str] = None,
        tag: Optional[str] = None,
        status: NotificationStatus = NotificationStatus.COMPLETED,
    ) -> "NotificationBuilder":
        """Set the content for the notification."""
        self.content = NotificationContent(title=title, message=message, icon=icon, tag=tag, status=status)
        return self

    def set_payload(
        self, *, category: NotificationCategory = NotificationCategory.INAPP, source: str = app_settings.source_topic
    ) -> "NotificationBuilder":
        """Set the payload for the notification."""
        self.payload = NotificationPayload(category=category, source=source, content=self.content)
        return self

    def set_notification_request(
        self, *, subscriber_ids: Union[str, List[str]], name: str = BUD_NOTIFICATION_WORKFLOW
    ) -> NotificationRequest:
        """Build the notification request."""
        self.notification_request = NotificationRequest(name=name, subscriber_ids=subscriber_ids, payload=self.payload)
        return self

    def build(self) -> NotificationRequest:
        """Build the notification request."""
        notification = self.notification_request
        self.reset()
        return notification

    def reset(self) -> "NotificationBuilder":
        """Reset the builder."""
        self.content = None
        self.payload = None
        self.notification_request = None
        return self


class NotificationService:
    """Service for sending notifications."""

    def __init__(self):
        """Initialize the notification service."""
        self.notification_endpoint = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_notify_app_id}/method/notifications"
        )

    async def send_notification(self, notification: NotificationRequest) -> dict:
        """Send a notification.

        Args:
            notification (NotificationRequest): The notification to send

        Returns:
            dict: The response from the notification service
        """
        payload = notification.model_dump(exclude_none=True, mode="json")
        logger.debug(f"Sending notification with payload {payload}")

        try:
            async with aiohttp.ClientSession() as session, session.post(
                self.notification_endpoint, json=payload
            ) as response:
                response_data = await response.json()
                if response.status != 200:
                    logger.error(f"Failed to send notification: {response.status} {response_data}")

                logger.debug("Successfully sent notification")
                return response_data
        except Exception as e:
            logger.exception(f"Failed to send notification: {e}")


if __name__ == "__main__":
    notification = (
        NotificationBuilder()
        .set_content(title="example test", message="Test 3", icon="test")
        .set_payload()
        .set_notification_request(subscriber_ids=["3c0d43d8-5d11-4549-966d-1492c9063638"], name="bud-notification")
        .build()
    )
    import asyncio

    asyncio.run(NotificationService().send_notification(notification))

    # python -m budapp.commons.notification_utils
