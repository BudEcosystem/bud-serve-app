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
#  ----------------

"""The eval ops package, containing essential business logic, services, and routing configurations for the eval ops."""


import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.commons.dependencies import get_current_active_user, get_session
from budapp.commons.exceptions import ClientException
from budapp.commons.schemas import ErrorResponse
from budapp.eval_ops.schemas import (
    CreateEvaluationRequest,
    CreateEvaluationResponse,
    DeleteEvaluationResponse,
    ListEvaluationsResponse,
    UpdateEvaluationRequest,
    UpdateEvaluationResponse,
)
from budapp.eval_ops.schemas import Evaluation as EvaluationSchema
from budapp.eval_ops.services import EvaluationService
from budapp.user_ops.models import User


logger = logging.get_logger(__name__)

eval_router =APIRouter(prefix="/eval", tags=["eval"])


@eval_router.post(
    "/",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
        status.HTTP_201_CREATED: {
            "model": CreateEvaluationResponse,
            "description": "Successfully created evaluation",
        },
    },
    description="Create evaluation",
)
async def create_evaluation(
    request: CreateEvaluationRequest,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> CreateEvaluationResponse:
    """Create evaluation."""
    logger.debug(f"Creating evaluation: {request}")

    try:
        evaluation = await EvaluationService(session).create_evaluation(request, current_user)
    except ClientException as e:
        logger.error(f"Failed to create evaluation: {e.message}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to create evaluation: {e}")
        return ErrorResponse(code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to create evaluation").to_http_response()

    return CreateEvaluationResponse(
        code=status.HTTP_201_CREATED,
        object="evaluation.create",
        message="Successfully created evaluation",
        evaluation=EvaluationSchema(
                evaluation_id=evaluation.id,
                name=evaluation.name,
                description=evaluation.description,
                project_id=evaluation.project_id,
            ),
    )


@eval_router.get(
    "/",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_200_OK: {
            "model": ListEvaluationsResponse,
            "description": "Successfully listed evaluations",
        },
    },
    description="List evaluations",
)
async def list_evaluations(
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> ListEvaluationsResponse:
    """List evaluations."""
    logger.debug(f"Listing evaluations for user: {current_user.id}")
    evaluations = await EvaluationService(session).list_evaluations(current_user)
    return ListEvaluationsResponse(
        code=status.HTTP_200_OK,
        object="evaluation.list",
        message="Successfully listed evaluations",
        evaluations=evaluations,
    )

@eval_router.patch(
    "/{evaluation_id}",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "Evaluation not found or access denied"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Service unavailable due to server error"},
        status.HTTP_200_OK: {"model": UpdateEvaluationResponse, "description": "Successfully updated evaluation"},
    },
    description="Update an existing evaluation",
)
async def update_evaluation(
    evaluation_id: uuid.UUID,
    request: UpdateEvaluationRequest,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> UpdateEvaluationResponse:
    """Endpoint to update an evaluation."""
    logger.debug(f"Received update for evaluation {evaluation_id}: {request}")
    try:
        evaluation = await EvaluationService(session).update_evaluation(evaluation_id, request, current_user)
    except ClientException as e:
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Unexpected error updating evaluation: {e}")
        return ErrorResponse(code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to update evaluation").to_http_response()

    return UpdateEvaluationResponse(
        code=status.HTTP_200_OK,
        object="evaluation.update",
        message="Successfully updated evaluation",
        evaluation=EvaluationSchema(
            evaluation_id=evaluation.id,
            name=evaluation.name,
            description=evaluation.description,
            project_id=evaluation.project_id,
        ),
    )

@eval_router.delete(
    "/{evaluation_id}",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "Evaluation not found or access denied"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Service unavailable due to server error"},
        status.HTTP_200_OK: {"model": DeleteEvaluationResponse, "description": "Successfully deleted evaluation"},
    },
    description="Delete an existing evaluation (soft delete)",
)
async def delete_evaluation(
    evaluation_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> DeleteEvaluationResponse:
    """Endpoint to delete an evaluation."""
    logger.debug(f"Received delete request for evaluation {evaluation_id}")
    try:
        await EvaluationService(session).delete_evaluation(evaluation_id, current_user)
    except ClientException as e:
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Unexpected error deleting evaluation: {e}")
        return ErrorResponse(code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to delete evaluation").to_http_response()

    return DeleteEvaluationResponse(
        code=status.HTTP_200_OK,
        object="evaluation.delete",
        message="Successfully deleted evaluation"
    )
