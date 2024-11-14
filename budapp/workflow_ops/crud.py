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

"""The crud package, containing essential business logic, services, and routing configurations for the workflow ops."""

from typing import List

from sqlalchemy import select

from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils

from .models import WorkflowStep as WorkflowStepModel


logger = logging.get_logger(__name__)


class WorkflowDataManager(DataManagerUtils):
    """Data manager for the Workflow model."""

    pass


class WorkflowStepDataManager(DataManagerUtils):
    """Data manager for the WorkflowStep model."""

    async def get_all_workflow_steps(self, filters: dict) -> List[WorkflowStepModel]:
        """Get all workflow steps from the database."""
        stmt = select(WorkflowStepModel).filter_by(**filters).order_by(WorkflowStepModel.step_number)
        return self.scalars_all(stmt)

    async def get_all_workflow_steps_by_data(self, data_key: str, workflow_id: str) -> List[WorkflowStepModel]:
        """Get all workflow steps from the database by data key and workflow id."""
        stmt = (
            select(WorkflowStepModel)
            .filter(
                WorkflowStepModel.data.op("->>")(data_key).isnot(None), WorkflowStepModel.workflow_id == workflow_id
            )
            .order_by(WorkflowStepModel.step_number)
        )
        return self.scalars_all(stmt)
