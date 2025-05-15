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

"""The eval ops services. Contains business logic for eval ops."""

import uuid
from typing import List

import sqlalchemy

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.eval_ops.models import Evaluation as EvaluationModel
from budapp.eval_ops.schemas import CreateEvaluationRequest, UpdateEvaluationRequest
from budapp.user_ops.schemas import User


logger = logging.get_logger(__name__)


class EvaluationService(SessionMixin):
    """The evaluation service."""

    async def create_evaluation(self, request: CreateEvaluationRequest, current_user: User) -> EvaluationModel:
        """Create a new evaluation."""
        logger.debug(f"uuid.UUIDting evaluation: {request}")
        evaluation = EvaluationModel(
            name=request.name,
            description=request.description,
            project_id=uuid.UUID(request.project_id),
            created_by=current_user.id,
            status="active"
        )

        try:
            self.session.add(evaluation)
            self.session.commit()
            self.session.refresh(evaluation)
        except sqlalchemy.exc.IntegrityError as e:
            logger.error(f"Failed to create evaluation due to IntegrityError: {str(e)}")
            self.session.rollback()
            raise ClientException("Failed to create evaluation due to IntegrityError") from e
        except Exception as e:
            logger.error(f"Failed to create evaluation due to unexpected error: {str(e)}")
            self.session.rollback()
            raise ClientException("Failed to create evaluation due to unexpected error") from e

        return evaluation

    async def get_evals_by_project_id(self, project_id: uuid.UUID) -> List[EvaluationModel]:
        """Get evaluations by project id."""
        logger.debug(f"Fetching evaluations for project_id: {project_id}")
        try:
            evaluations = (
                self.session
                    .query(EvaluationModel)
                    .filter(EvaluationModel.project_id == project_id)
                    .all()
            )
        except sqlalchemy.exc.SQLAlchemyError as e:
            logger.error(f"Failed to fetch evaluations for project {project_id}: {e}")
            raise ClientException(f"Failed to fetch evaluations for project {project_id}") from e
        return evaluations

    async def update_evaluation(
        self,
        evaluation_id: uuid.UUID,
        request: UpdateEvaluationRequest,
        current_user: User,
    ) -> EvaluationModel:
        """Update an existing evaluation."""
        logger.debug(f"Updating evaluation {evaluation_id} with data: {request}")
        evaluation = self.session.get(EvaluationModel, evaluation_id)
        # Ensure it exists, belongs to the user, and is not deleted
        if not evaluation or evaluation.created_by != current_user.id or evaluation.status == "deleted":
            raise ClientException("Evaluation not found or access denied", status_code=404)
        # Apply updates
        if request.name is not None:
            evaluation.name = request.name
        if request.description is not None:
            evaluation.description = request.description
        try:
            self.session.commit()
            self.session.refresh(evaluation)
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Failed to update evaluation {evaluation_id}: {e}")
            raise ClientException("Failed to update evaluation") from e
        return evaluation

    async def delete_evaluation(
        self,
        evaluation_id: uuid.UUID,
        current_user: User,
    ) -> None:
        """Delete (soft-delete) an existing evaluation."""
        logger.debug(f"Deleting evaluation {evaluation_id}")
        evaluation = self.session.get(EvaluationModel, evaluation_id)
        if not evaluation or evaluation.created_by != current_user.id or evaluation.status == "deleted":
            raise ClientException("Evaluation not found or access denied", status_code=404)
        # mark as deleted
        evaluation.status = "deleted"
        try:
            self.session.commit()
        except sqlalchemy.exc.SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Failed to delete evaluation {evaluation_id}: {e}")
            raise ClientException("Failed to delete evaluation") from e
