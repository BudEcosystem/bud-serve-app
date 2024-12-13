from typing import Any, Dict
from uuid import UUID

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException

from .crud import ProjectDataManager
from .models import Project as ProjectModel
from .schemas import ProjectResponse

logger = logging.get_logger(__name__)


class ProjectService(SessionMixin):
    """Project service."""

    async def edit_project(self, project_id: UUID, data: Dict[str, Any]) -> ProjectResponse:
        """Edit project by validating and updating specific fields."""
        # Retrieve existing model
        db_project = await ProjectDataManager(self.session).retrieve_by_fields(
            model=ProjectModel, fields={"id": project_id}
        )

        if "name" in data:
            duplicate_project = await ProjectDataManager(self.session).retrieve_by_fields(
                model=ProjectModel,
                fields={"name": data["name"], "is_active": True},
                exclude_fields={"id": project_id},
                missing_ok=True,
            )
            if duplicate_project:
                raise ClientException("Project name already exists")

        db_project = await ProjectDataManager(self.session).update_by_fields(db_project, data)

        return db_project
