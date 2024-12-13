from typing import Any, Dict
from uuid import UUID

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin

from .crud import ProjectDataManager
from .models import Project as ProjectModel
from .schemas import ProjectResponse

logger = logging.get_logger(__name__)


class ProjectService(SessionMixin):
    """Cluster service."""
    async def edit_project(self, project_id: UUID, data: Dict[str, Any]) -> ProjectResponse:
        """Edit project by validating and updating specific fields."""
        # Retrieve existing model
        db_project = await ProjectDataManager(self.session).retrieve_by_fields(
            model=ProjectModel, fields={"id": project_id}
        )
        if not db_project:
            raise ValueError(f"Project with ID {project_id} not found")

        updated_project = await ProjectDataManager(self.session).update_by_fields(db_project, data)
        updated_project = ProjectResponse.model_validate(updated_project)

        return updated_project