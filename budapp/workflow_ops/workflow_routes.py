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

"""The workflow ops package, containing essential business logic, services, and routing configurations for the workflow ops."""

from typing import Union
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons import logging
from budapp.commons.dependencies import get_current_active_user, get_session
from budapp.commons.exceptions import ClientException
from budapp.commons.schemas import ErrorResponse
from budapp.user_ops.schemas import User

from .schemas import RetrieveWorkflowDataResponse
from .services import WorkflowService


logger = logging.get_logger(__name__)

workflow_router = APIRouter(prefix="/workflows", tags=["workflow"])


@workflow_router.get(
    "/{workflow_id}",
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
            "description": "Successfully retrieve workflow data",
        },
    },
    description="Retrieve workflow data",
)
async def retrieve_workflow_data(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    workflow_id: UUID,
) -> Union[RetrieveWorkflowDataResponse, ErrorResponse]:
    """Retrieve workflow data."""
    try:
        return await WorkflowService(session).retrieve_workflow_data(workflow_id)
    except ClientException as e:
        logger.exception(f"Failed to retrieve workflow data: {e}")
        return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to retrieve workflow data: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to retrieve workflow data"
        ).to_http_response()
