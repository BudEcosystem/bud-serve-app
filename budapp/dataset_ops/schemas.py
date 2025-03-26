from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from budapp.commons.schemas import PaginatedSuccessResponse


class DatasetResponse(BaseModel):
    id: UUID
    name: str
    tags: list
    num_of_prompts: int


class DatasetPaginatedResponse(PaginatedSuccessResponse):
    datasets: list[DatasetResponse]


class DatasetFilter(BaseModel):
    name: Optional[str] = None
    tags: Optional[str] = None
