# budapp/eval_ops/routers.py

"""Evaluation Operations API Routes.

This module defines the FastAPI routes for creating, listing, updating,
and deleting evaluations, as well as listing evaluation traits. Each endpoint
is documented with a docstring specifying its purpose, parameters, and response.
"""

import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from budapp.commons.dependencies import get_current_active_user, get_session
from budapp.commons.schemas import ErrorResponse
from budapp.eval_ops.schemas import (
    CreateEvaluationRequest,
    CreateEvaluationResponse,
    CreateRunRequest,
    CreateRunResponse,
    DeleteEvaluationResponse,
    DeleteRunResponse,
    ListEvaluationsResponse,
    ListRunsResponse,
    ListTraitsResponse,
    UpdateEvaluationRequest,
    UpdateEvaluationResponse,
    UpdateRunRequest,
    UpdateRunResponse,
)
from budapp.eval_ops.services import EvaluationService
from budapp.user_ops.models import User


router = APIRouter(prefix="/eval", tags=["eval"])


@router.post(
    "/",
    response_model=CreateEvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def create_evaluation(
    request: Annotated[CreateEvaluationRequest, Depends()],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Create a new evaluation.

    - **request**: Payload containing `name`, `description`, and `project_id`.
    - **session**: Database session dependency.
    - **current_user**: The authenticated user creating the evaluation.

    Returns a `CreateEvaluationResponse` with the created evaluation.
    """
    try:
        ev = EvaluationService(session).create_evaluation(request, current_user.id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create evaluation") from e
    return CreateEvaluationResponse(
        code=status.HTTP_201_CREATED,
        object="evaluation.create",
        message="Successfully created evaluation",
        evaluation=ev,
    )


@router.get(
    "/",
    response_model=ListEvaluationsResponse,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse}},
)
def list_evaluations(
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """List all evaluations for the current user.

    - **session**: Database session dependency.
    - **current_user**: The authenticated user whose evaluations are listed.

    Returns a `ListEvaluationsResponse` containing a list of evaluations.
    """
    try:
        evs = EvaluationService(session).list_evaluations(current_user.id)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list evaluations") from e
    return ListEvaluationsResponse(
        code=status.HTTP_200_OK,
        object="evaluation.list",
        message="Successfully listed evaluations",
        evaluations=evs,
    )


@router.patch(
    "/{evaluation_id}",
    response_model=UpdateEvaluationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def update_evaluation(
    evaluation_id: Annotated[uuid.UUID, Path(..., description="ID of evaluation to update")],
    request: Annotated[UpdateEvaluationRequest, Depends()],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Update an existing evaluation.

    - **evaluation_id**: UUID of the evaluation to update.
    - **request**: Payload with optional `name` and/or `description` fields.
    - **session**: Database session dependency.
    - **current_user**: The authenticated user performing the update.

    Returns an `UpdateEvaluationResponse` with the updated evaluation.
    """
    try:
        ev = EvaluationService(session).update_evaluation(evaluation_id, request, current_user.id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update evaluation") from e
    return UpdateEvaluationResponse(
        code=status.HTTP_200_OK,
        object="evaluation.update",
        message="Successfully updated evaluation",
        evaluation=ev,
    )


@router.delete(
    "/{evaluation_id}",
    response_model=DeleteEvaluationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def delete_evaluation(
    evaluation_id: Annotated[uuid.UUID, Path(..., description="ID of evaluation to delete")],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Soft-delete an existing evaluation by marking its status as 'deleted'.

    - **evaluation_id**: UUID of the evaluation to delete.
    - **session**: Database session dependency.
    - **current_user**: The authenticated user performing the deletion.

    Returns a `DeleteEvaluationResponse` confirming the deletion.
    """
    try:
        EvaluationService(session).delete_evaluation(evaluation_id, current_user.id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete evaluation") from e
    return DeleteEvaluationResponse(
        code=status.HTTP_200_OK,
        object="evaluation.delete",
        message="Successfully deleted evaluation",
    )


@router.get(
    "/traits",
    response_model=ListTraitsResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def list_traits(
    page: Annotated[int, Query(1, ge=1, description="Page number")],
    limit: Annotated[int, Query(10, ge=1, description="Results per page")],
    name: Annotated[Optional[str], Query(None, description="Filter by trait name")],
    unique_id: Annotated[Optional[str], Query(None, description="Filter by trait UUID")],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """List evaluation traits with optional filtering and pagination.

    - **page**: Page number (default: 1).
    - **limit**: Number of traits per page (default: 10).
    - **name**: Optional case-insensitive substring filter on trait name.
    - **unique_id**: Optional exact UUID filter on trait ID.
    - **session**: Database session dependency.
    - **current_user**: The authenticated user (unused in this endpoint).

    Returns a `ListTraitsResponse` containing traits, total count, and pagination info.
    """
    offset = (page - 1) * limit
    try:
        traits, total = EvaluationService(session).list_traits(offset, limit, name, unique_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list traits") from e
    return ListTraitsResponse(
        code=status.HTTP_200_OK,
        object="traits.list",
        message="Successfully listed traits",
        traits=traits,
        total_record=total,
        page=page,
        limit=limit,
    )

@router.post(
    "/{evaluation_id}/runs",
    response_model=CreateRunResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def create_run(
    evaluation_id: Annotated[uuid.UUID, Path(..., description="Evaluation ID")],
    request: Annotated[CreateRunRequest, Depends()],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Create a new run under the specified evaluation."""
    run = EvaluationService(session).create_run(evaluation_id, request, current_user.id)
    return CreateRunResponse(
        code=status.HTTP_201_CREATED,
        object="run.create",
        message="Successfully created run",
        run=run,
    )


@router.get(
    "/{evaluation_id}/runs",
    response_model=ListRunsResponse,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
def list_runs(
    evaluation_id: Annotated[uuid.UUID, Path(..., description="Evaluation ID")],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """List all runs for the specified evaluation."""
    runs = EvaluationService(session).list_runs(evaluation_id, current_user.id)
    return ListRunsResponse(
        code=status.HTTP_200_OK,
        object="run.list",
        message="Successfully listed runs",
        runs=runs,
    )


@router.patch(
    "/runs/{run_id}",
    response_model=UpdateRunResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def update_run(
    run_id: Annotated[uuid.UUID, Path(..., description="Run ID")],
    request: Annotated[UpdateRunRequest, Depends()],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Update an existing run's metadata or status."""
    run = EvaluationService(session).update_run(run_id, request, current_user.id)
    return UpdateRunResponse(
        code=status.HTTP_200_OK,
        object="run.update",
        message="Successfully updated run",
        run=run,
    )


@router.delete(
    "/runs/{run_id}",
    response_model=DeleteRunResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def delete_run(
    run_id: Annotated[uuid.UUID, Path(..., description="Run ID")],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Soft-delete (cancel) a run by marking its status DELETED."""
    EvaluationService(session).delete_run(run_id, current_user.id)
    return DeleteRunResponse(
        code=status.HTTP_200_OK,
        object="run.delete",
        message="Successfully deleted run",
    )
