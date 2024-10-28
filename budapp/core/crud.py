from typing import List

from sqlalchemy import select

from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils
from budapp.core.models import WorkflowStep


logger = logging.get_logger(__name__)


class WorkflowDataManager(DataManagerUtils):
    """Data manager for the Workflow model."""

    pass


class WorkflowStepDataManager(DataManagerUtils):
    """Data manager for the WorkflowStep model."""

    async def get_all_workflow_steps(self, filters: dict) -> List[WorkflowStep]:
        """Get all workflow steps from the database."""
        stmt = select(WorkflowStep).filter_by(**filters)
        return self.scalars_all(stmt)
