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

"""The crud package, containing essential business logic, services, and routing configurations for the microservices."""

from typing import List, Tuple, Dict

from sqlalchemy import select

from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils

from .models import WorkflowStep as WorkflowStepModel, ModelTemplate
from sqlalchemy import func, or_, select


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


class IconDataManager(DataManagerUtils):
    """Data manager for the Icon model."""

    pass

class ModelTemplateDataManager(DataManagerUtils):
    """Model template data manager class responsible for operations over database."""

    async def get_all_model_templates(
        self,
        offset: int,
        limit: int,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[ModelTemplate], int]:
        """List all model templates in the database."""

        # Validate filter fields
        await self.validate_fields(ModelTemplate, filters)

        # Generate statements according to search or filters
        if search:
            search_conditions = await self.generate_search_stmt(ModelTemplate, filters)
            stmt = select(
                ModelTemplate,
            ).filter(or_(*search_conditions))
            count_stmt = (
                select(func.count())
                .select_from(ModelTemplate)
                .filter(or_(*search_conditions))
            )
        else:
            stmt = select(
                ModelTemplate,
            ).filter_by(**filters)
            count_stmt = (
                select(func.count()).select_from(ModelTemplate).filter_by(**filters)
            )

        # Calculate count before applying limit and offset
        count = self.execute_scalar(count_stmt)

        # Apply limit and offset
        stmt = stmt.limit(limit).offset(offset)

        # Apply sorting
        if order_by:
            sort_conditions = await self.generate_sorting_stmt(ModelTemplate, order_by)
            stmt = stmt.order_by(*sort_conditions)

        result = self.scalars_all(stmt)

        return result, count