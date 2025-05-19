# budapp/eval_ops/services.py

"""EvaluationService.

This module contains the `EvaluationService` class which encapsulates
business logic for creating, listing, updating, and deleting
Evaluation records, as well as listing Traits for pagination.

Each method raises `HTTPException` with appropriate status codes
for error conditions, and returns Pydantic schemas for successful operations.
"""

import uuid
from typing import List, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from budapp.eval_ops.models import Evaluation as EvaluationModel
from budapp.eval_ops.models import (
    EvaluationRunStatusEnum,
)
from budapp.eval_ops.models import (
    Run as RunModel,
)
from budapp.eval_ops.models import Trait as TraitModel
from budapp.eval_ops.schemas import (
    CreateEvaluationRequest,
    CreateRunRequest,
    UpdateEvaluationRequest,
    UpdateRunRequest,
)
from budapp.eval_ops.schemas import (
    Evaluation as EvaluationSchema,
)
from budapp.eval_ops.schemas import (
    Run as RunSchema,
)
from budapp.eval_ops.schemas import (
    Trait as TraitSchema,
)


class EvaluationService:
    """Service layer for Evaluation operations.

    Methods:
        - create_evaluation: create and persist a new Evaluation.
        - list_evaluations: retrieve all non-deleted Evaluations for a user.
        - update_evaluation: apply updates to an existing Evaluation.
        - delete_evaluation: perform a soft delete on an Evaluation.
        - list_traits: list Trait entries with optional filters and pagination.
    """

    def __init__(self, session: Session):
        """Initialize the service with a database session.

        Parameters:
            session (Session): SQLAlchemy database session.
        """
        self.session = session

    def create_evaluation(self, req: CreateEvaluationRequest, user_id: uuid.UUID) -> EvaluationSchema:
        """Create a new Evaluation record.

        Parameters:
            req (CreateEvaluationRequest): Payload containing name, description, project_id.
            user_id (uuid.UUID): ID of the user creating the evaluation.

        Returns:
            EvaluationSchema: Pydantic schema of the created Evaluation.

        Raises:
            HTTPException(status_code=500): If database insertion fails.
        """
        ev = EvaluationModel(
            name=req.name,
            description=req.description,
            project_id=uuid.UUID(req.project_id),
            created_by=user_id,
            status="active",
        )
        try:
            self.session.add(ev)
            self.session.commit()
            self.session.refresh(ev)
        except Exception as e:
            self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create evaluation"
            ) from e
        return EvaluationSchema.from_orm(ev)

    def list_evaluations(self, user_id: uuid.UUID) -> List[EvaluationSchema]:
        """List all non-deleted Evaluations for a given user.

        Parameters:
            user_id (uuid.UUID): ID of the user whose evaluations to list.

        Returns:
            List[EvaluationSchema]: List of Pydantic schemas for each Evaluation.

        Raises:
            HTTPException(status_code=500): If database query fails.
        """
        try:
            q = (
                self.session.query(EvaluationModel)
                .filter(
                    EvaluationModel.created_by == user_id,
                    EvaluationModel.status != "deleted",
                )
                .order_by(EvaluationModel.created_at.desc())
            )
            evs = q.all()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list evaluations"
            ) from e
        return [EvaluationSchema.from_orm(e) for e in evs]

    def update_evaluation(
        self,
        ev_id: uuid.UUID,
        req: UpdateEvaluationRequest,
        user_id: uuid.UUID,
    ) -> EvaluationSchema:
        """Update fields of an existing Evaluation.

        Parameters:
            ev_id (uuid.UUID): ID of the evaluation to update.
            req (UpdateEvaluationRequest): Payload with optional name/description.
            user_id (uuid.UUID): ID of the user attempting the update.

        Returns:
            EvaluationSchema: Pydantic schema of the updated Evaluation.

        Raises:
            HTTPException(status_code=404): If evaluation not found or access denied.
            HTTPException(status_code=500): If database update fails.
        """
        ev = self.session.get(EvaluationModel, ev_id)
        if not ev or ev.created_by != user_id or ev.status == "deleted":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found or access denied"
            )
        if req.name is not None:
            ev.name = req.name
        if req.description is not None:
            ev.description = req.description
        try:
            self.session.commit()
            self.session.refresh(ev)
        except Exception as e:
            self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update evaluation"
            ) from e
        return EvaluationSchema.from_orm(ev)

    def delete_evaluation(self, ev_id: uuid.UUID, user_id: uuid.UUID) -> None:
        """Soft-delete an Evaluation by setting its status to 'deleted'.

        Parameters:
            ev_id (uuid.UUID): ID of the evaluation to delete.
            user_id (uuid.UUID): ID of the user attempting the delete.

        Raises:
            HTTPException(status_code=404): If evaluation not found or access denied.
            HTTPException(status_code=500): If database commit fails.
        """
        ev = self.session.get(EvaluationModel, ev_id)
        if not ev or ev.created_by != user_id or ev.status == "deleted":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found or access denied"
            )
        ev.status = "deleted"
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete evaluation"
            ) from e

    def list_traits(
        self,
        offset: int = 0,
        limit: int = 10,
        name: Optional[str] = None,
        unique_id: Optional[str] = None,
    ) -> Tuple[List[TraitSchema], int]:
        """Paginate and filter Trait records.

        Parameters:
            offset (int): Number of records to skip (for pagination).
            limit (int): Maximum number of records to return.
            name (Optional[str]): Case-insensitive substring filter on trait name.
            unique_id (Optional[str]): Exact UUID filter on trait ID.

        Returns:
            Tuple[List[TraitSchema], int]: A tuple of (list of TraitSchema, total count).

        Raises:
            HTTPException(status_code=500): If database query fails.
        """
        try:
            q = self.session.query(TraitModel)
            if name:
                q = q.filter(TraitModel.name.ilike(f"%{name}%"))
            if unique_id:
                q = q.filter(TraitModel.id == uuid.UUID(unique_id))
            total = q.count()
            traits = (
                q.order_by(TraitModel.created_at)
                 .offset(offset)
                 .limit(limit)
                 .all()
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list traits"
            ) from e

        result = [
            TraitSchema(
                unique_id=str(t.id),
                name=t.name,
                description=t.description or "",
                category="",
                evals_ids=[],
            )
            for t in traits
        ]
        return result, total
    def create_run(self, evaluation_id: uuid.UUID, req: CreateRunRequest, user_id: uuid.UUID) -> RunSchema:
        """Create a new run under the given evaluation.

        Raises 404 if evaluation not found or access denied.
        """
        ev = self.session.get(EvaluationModel, evaluation_id)
        if not ev or ev.created_by != user_id or ev.status == EvaluationRunStatusEnum.DELETED.value:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Evaluation not found or access denied")
        run = RunModel(
            evaluation_id=evaluation_id,
            name=req.name,
            description=req.description,
            status=EvaluationRunStatusEnum.PENDING.value,
        )
        try:
            self.session.add(run)
            self.session.commit()
            self.session.refresh(run)
        except Exception as e:
            self.session.rollback()
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to create run") from e
        return RunSchema.from_orm(run)

    def list_runs(self, evaluation_id: uuid.UUID, user_id: uuid.UUID) -> List[RunSchema]:
        """List all runs for a given evaluation.

        Raises 404 if evaluation not found or access denied.
        """
        ev = self.session.get(EvaluationModel, evaluation_id)
        if not ev or ev.created_by != user_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Evaluation not found or access denied")
        runs = (
            self.session.query(RunModel)
            .filter(RunModel.evaluation_id == evaluation_id)
            .order_by(RunModel.created_at.desc())
            .all()
        )
        return [RunSchema.from_orm(r) for r in runs]

    def update_run(
        self,
        run_id: uuid.UUID,
        req: UpdateRunRequest,
        user_id: uuid.UUID,
    ) -> RunSchema:
        """Update fields of an existing run. 404 if not found/access denied.

        Raises 404 if run not found.
        """
        run = self.session.get(RunModel, run_id)
        if not run:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
        # check ownership via parent evaluation
        if run.evaluation.created_by != user_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Access denied")
        if req.name is not None:
            run.name = req.name
        if req.description is not None:
            run.description = req.description
        if req.status is not None:
            run.status = req.status.value
        try:
            self.session.commit()
            self.session.refresh(run)
        except Exception as e:
            self.session.rollback()
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to update run") from e
        return RunSchema.from_orm(run)

    def delete_run(self, run_id: uuid.UUID, user_id: uuid.UUID) -> None:
        """Soft-delete a run by marking its status as DELETED. 404 if not found/access denied.

        Raises 404 if run not found.
        """
        run = self.session.get(RunModel, run_id)
        if not run or run.evaluation.created_by != user_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found or access denied")
        run.status = EvaluationRunStatusEnum.DELETED.value
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to delete run") from e
