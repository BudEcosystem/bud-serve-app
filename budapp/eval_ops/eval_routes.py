import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.commons.dependencies import get_current_active_user, get_session
from budapp.commons.schemas import ErrorResponse
from budapp.eval_ops.schemas import (
    CreateExperimentRequest,
    CreateExperimentResponse,
    CreateRunRequest,
    CreateRunResponse,
    DeleteExperimentResponse,
    DeleteRunResponse,
    GetDatasetResponse,
    ListDatasetsResponse,
    CreateDatasetRequest,
    CreateDatasetResponse,
    UpdateDatasetRequest,
    UpdateDatasetResponse,
    DeleteDatasetResponse,
    DatasetFilter,
    ListExperimentsResponse,
    ListRunsResponse,
    ListTraitsResponse,
    UpdateExperimentRequest,
    UpdateExperimentResponse,
    UpdateRunRequest,
    UpdateRunResponse,
)
from budapp.eval_ops.services import ExperimentService
from budapp.user_ops.models import User


router = APIRouter(prefix="/experiments", tags=["experiments"])

logger = logging.get_logger(__name__)

@router.post(
    "/",
    response_model=CreateExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def create_experiment(
    request: Annotated[CreateExperimentRequest, Depends()],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Create a new experiment.

    - **request**: Payload containing `name`, `description`, and `project_id`.
    - **session**: Database session dependency.
    - **current_user**: The authenticated user creating the experiment.

    Returns a `CreateExperimentResponse` with the created experiment.
    """
    try:
        ev = ExperimentService(session).create_experiment(request, current_user.id)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.debug(f"Failed to create experiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create experiment") from e
    return CreateExperimentResponse(
        code=status.HTTP_201_CREATED,
        object="experiment.create",
        message="Successfully created experiment",
        experiment=ev,
    )


@router.get(
    "/",
    response_model=ListExperimentsResponse,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse}},
)
def list_experiments(
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """List all experiments for the current user.

    - **session**: Database session dependency.
    - **current_user**: The authenticated user whose experiments are listed.

    Returns a `ListExperimentsResponse` containing a list of experiments.
    """
    try:
        evs = ExperimentService(session).list_experiments(current_user.id)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list experiments") from e
    return ListExperimentsResponse(
        code=status.HTTP_200_OK,
        object="experiment.list",
        message="Successfully listed experiments",
        experiments=evs,
    )


@router.patch(
    "/{experiment_id}",
    response_model=UpdateExperimentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def update_experiment(
    experiment_id: Annotated[uuid.UUID, Path(..., description="ID of experiment to update")],
    request: Annotated[UpdateExperimentRequest, Depends()],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Update an existing experiment.

    - **experiment_id**: UUID of the experiment to update.
    - **request**: Payload with optional `name` and/or `description` fields.
    - **session**: Database session dependency.
    - **current_user**: The authenticated user performing the update.

    Returns an `UpdateExperimentResponse` with the updated experiment.
    """
    try:
        ev = ExperimentService(session).update_experiment(experiment_id, request, current_user.id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update experiment") from e
    return UpdateExperimentResponse(
        code=status.HTTP_200_OK,
        object="experiment.update",
        message="Successfully updated experiment",
        experiment=ev,
    )


@router.delete(
    "/{experiment_id}",
    response_model=DeleteExperimentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def delete_experiment(
    experiment_id: Annotated[uuid.UUID, Path(..., description="ID of experiment to delete")],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Soft-delete an existing experiment by marking its status as 'deleted'.

    - **experiment_id**: UUID of the experiment to delete.
    - **session**: Database session dependency.
    - **current_user**: The authenticated user performing the deletion.

    Returns a `DeleteExperimentResponse` confirming the deletion.
    """
    try:
        ExperimentService(session).delete_experiment(experiment_id, current_user.id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete experiment") from e
    return DeleteExperimentResponse(
        code=status.HTTP_200_OK,
        object="experiment.delete",
        message="Successfully deleted experiment",
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
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    limit: Annotated[int, Query(ge=1, description="Results per page")] = 10,
    name: Annotated[Optional[str], Query(description="Filter by trait name")] = None,
    unique_id: Annotated[Optional[str], Query(description="Filter by trait UUID")] = None,
):
    """List experiment traits with optional filtering and pagination.

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
        traits, total = ExperimentService(session).list_traits(offset, limit, name, unique_id)
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


@router.get(
    "/datasets/{dataset_id}",
    response_model=GetDatasetResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def get_dataset_by_id(
    dataset_id: Annotated[uuid.UUID, Path(..., description="ID of dataset to retrieve")],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Get an evaluation dataset by ID with associated traits.

    - **dataset_id**: UUID of the dataset to retrieve.
    - **session**: Database session dependency.
    - **current_user**: The authenticated user (for access control).

    Returns a `GetDatasetResponse` containing the dataset with traits information.
    """
    try:
        dataset = ExperimentService(session).get_dataset_by_id(dataset_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get dataset") from e
    return GetDatasetResponse(
        code=status.HTTP_200_OK,
        object="dataset.get",
        message="Successfully retrieved dataset",
        dataset=dataset,
    )


@router.post(
    "/{experiment_id}/runs",
    response_model=CreateRunResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def create_run(
    experiment_id: Annotated[uuid.UUID, Path(..., description="Experiment ID")],
    request: Annotated[CreateRunRequest, Depends()],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Create a new run under the specified experiment."""
    run = ExperimentService(session).create_run(experiment_id, request, current_user.id)
    return CreateRunResponse(
        code=status.HTTP_201_CREATED,
        object="run.create",
        message="Successfully created run",
        run=run,
    )


@router.get(
    "/{experiment_id}/runs",
    response_model=ListRunsResponse,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
def list_runs(
    experiment_id: Annotated[uuid.UUID, Path(..., description="Experiment ID")],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """List all runs for the specified experiment."""
    runs = ExperimentService(session).list_runs(experiment_id, current_user.id)
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
    run = ExperimentService(session).update_run(run_id, request, current_user.id)
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
    ExperimentService(session).delete_run(run_id, current_user.id)
    return DeleteRunResponse(
        code=status.HTTP_200_OK,
        object="run.delete",
        message="Successfully deleted run",
    )


@router.get(
    "/datasets",
    response_model=ListDatasetsResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def list_datasets(
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    limit: Annotated[int, Query(ge=1, description="Results per page")] = 10,
    name: Annotated[Optional[str], Query(description="Filter by dataset name")] = None,
    modalities: Annotated[Optional[str], Query(description="Filter by modalities (comma-separated)")] = None,
    language: Annotated[Optional[str], Query(description="Filter by languages (comma-separated)")] = None,
    domains: Annotated[Optional[str], Query(description="Filter by domains (comma-separated)")] = None,
):
    """List evaluation datasets with optional filtering and pagination.

    - **page**: Page number (default: 1).
    - **limit**: Number of datasets per page (default: 10).
    - **name**: Optional case-insensitive substring filter on dataset name.
    - **modalities**: Optional comma-separated list of modalities to filter by.
    - **language**: Optional comma-separated list of languages to filter by.
    - **domains**: Optional comma-separated list of domains to filter by.
    - **session**: Database session dependency.
    - **current_user**: The authenticated user.

    Returns a `ListDatasetsResponse` containing datasets with traits, total count, and pagination info.
    """
    offset = (page - 1) * limit
    
    # Parse comma-separated filters
    filters = DatasetFilter(
        name=name,
        modalities=modalities.split(",") if modalities else None,
        language=language.split(",") if language else None,
        domains=domains.split(",") if domains else None,
    )
    
    try:
        datasets, total = ExperimentService(session).list_datasets(offset, limit, filters)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list datasets") from e
    
    return ListDatasetsResponse(
        code=status.HTTP_200_OK,
        object="datasets.list",
        message="Successfully listed datasets",
        datasets=datasets,
        total_record=total,
        page=page,
        limit=limit,
    )


@router.post(
    "/datasets",
    response_model=CreateDatasetResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def create_dataset(
    request: Annotated[CreateDatasetRequest, Depends()],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Create a new evaluation dataset.

    - **request**: Payload containing dataset information and trait associations.
    - **session**: Database session dependency.
    - **current_user**: The authenticated user creating the dataset.

    Returns a `CreateDatasetResponse` with the created dataset and traits.
    """
    try:
        dataset = ExperimentService(session).create_dataset(request)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.debug(f"Failed to create dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create dataset") from e
    
    return CreateDatasetResponse(
        code=status.HTTP_201_CREATED,
        object="dataset.create",
        message="Successfully created dataset",
        dataset=dataset,
    )


@router.patch(
    "/datasets/{dataset_id}",
    response_model=UpdateDatasetResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def update_dataset(
    dataset_id: Annotated[uuid.UUID, Path(..., description="ID of dataset to update")],
    request: Annotated[UpdateDatasetRequest, Depends()],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Update an existing evaluation dataset.

    - **dataset_id**: UUID of the dataset to update.
    - **request**: Payload with optional dataset fields and trait associations.
    - **session**: Database session dependency.
    - **current_user**: The authenticated user performing the update.

    Returns an `UpdateDatasetResponse` with the updated dataset and traits.
    """
    try:
        dataset = ExperimentService(session).update_dataset(dataset_id, request)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update dataset") from e
    
    return UpdateDatasetResponse(
        code=status.HTTP_200_OK,
        object="dataset.update",
        message="Successfully updated dataset",
        dataset=dataset,
    )


@router.delete(
    "/datasets/{dataset_id}",
    response_model=DeleteDatasetResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
def delete_dataset(
    dataset_id: Annotated[uuid.UUID, Path(..., description="ID of dataset to delete")],
    session: Annotated[Session, Depends(get_session)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    """Delete an evaluation dataset and its trait associations.

    - **dataset_id**: UUID of the dataset to delete.
    - **session**: Database session dependency.
    - **current_user**: The authenticated user performing the deletion.

    Returns a `DeleteDatasetResponse` confirming the deletion.
    """
    try:
        ExperimentService(session).delete_dataset(dataset_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete dataset") from e
    
    return DeleteDatasetResponse(
        code=status.HTTP_200_OK,
        object="dataset.delete",
        message="Successfully deleted dataset",
    )
