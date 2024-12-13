from budapp.commons.schemas import SuccessResponse, Tag
from typing import List
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class ProjectBase(BaseModel):
    name: str
    description: str | None = None
    tags: List[Tag] | None = None
    icon: str | None = None


class EditProjectRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = Field(None, max_length=300)
    tags: List[Tag] | None = None
    icon: str | None = None

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        """Ensure the name is not empty or only whitespace."""
        if value is not None and not value.strip():
            raise ValueError("Project name cannot be empty or only whitespace.")
        return value


class ProjectResponse(ProjectBase):
    """Project response to client schema"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID4


class SingleProjectResponse(SuccessResponse):
    project: ProjectResponse
