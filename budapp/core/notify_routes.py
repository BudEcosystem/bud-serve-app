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

"""Defines metadata routes for the microservices, providing endpoints for retrieving service-level information."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.commons.api_utils import pubsub_api_endpoint
from budapp.commons.constants import NotificationCategory, PayloadType
from budapp.commons.dependencies import get_session
from budapp.commons.exceptions import ClientException
from budapp.commons.schemas import ErrorResponse

from .schemas import NotificationRequest, NotificationResponse
from .services import NotificationService


logger = logging.get_logger(__name__)

notify_router = APIRouter()


@notify_router.post(
    "/notifications",
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
            "model": NotificationResponse,
            "description": "Successfully triggered notification",
        },
    },
    status_code=status.HTTP_200_OK,
    description="Triggers a notification. Can be used for both API and PubSub. Refer to NotificationRequest schema for details.",
    tags=["Notifications"],
)
@pubsub_api_endpoint(NotificationRequest)
async def receive_notification(
    notification: NotificationRequest,
    session: Annotated[Session, Depends(get_session)],
) -> Response:
    """Receives a notification.

    This method interacts with other microservices to receive notifications.

    Args:
        notification (NotificationRequest): The request object containing notification details
            such as payload.

    Returns:
        Response: A response object containing the status of the notification receiving
        process and related information.
    """
    logger.debug("Received request to subscribe to bud-serve-app notifications")
    try:
        logger.info("Subscribed to bud-serve-app notifications successfully")
        payload = notification.payload

        # Check if the notification is internal
        if payload.category == NotificationCategory.INTERNAL and payload.type == PayloadType.DEPLOYMENT_RECOMMENDATION:
            await NotificationService(session).update_recommended_cluster_events(payload)
            return NotificationResponse(
                object="notification",
                message="Updated recommended cluster event in workflow step",
            ).to_http_response()
        if payload.category == NotificationCategory.INTERNAL and payload.type == PayloadType.DEPLOY_MODEL:
            await NotificationService(session).update_model_deployment_events(payload)
            return NotificationResponse(
                object="notification",
                message="Updated model deployment event in workflow step",
            ).to_http_response()
        if payload.category == NotificationCategory.INTERNAL and payload.type == PayloadType.REGISTER_CLUSTER:
            await NotificationService(session).update_cluster_creation_events(payload)
            return NotificationResponse(
                object="notification",
                message="Updated cluster creation event in workflow step",
            ).to_http_response()
        if payload.category == NotificationCategory.INTERNAL and payload.type == PayloadType.PERFORM_MODEL_EXTRACTION:
            await NotificationService(session).update_model_extraction_events(payload)
            return NotificationResponse(
                object="notification",
                message="Updated model extraction event in workflow step",
            ).to_http_response()
        if (
            payload.category == NotificationCategory.INTERNAL
            and payload.type == PayloadType.PERFORM_MODEL_SECURITY_SCAN
        ):
            await NotificationService(session).update_model_security_scan_events(payload)
            return NotificationResponse(
                object="notification",
                message="Updated model security scan event in workflow step",
            ).to_http_response()
        else:
            return NotificationResponse(
                object="notification",
                message="Pubsub notification received",
            ).to_http_response()
    except ClientException as e:
        logger.exception(f"Failed to execute notification: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as err:
        logger.exception(f"Unexpected error occurred while receiving notification. {err}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            type="InternalServerError",
            message="Unexpected error occurred while receiving notification.",
        )
