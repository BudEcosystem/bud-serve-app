from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from budapp.commons.schemas import PaginatedSuccessResponse

from ..commons.constants import DatasetStatusEnum


class DatasetResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    tags: Optional[dict] = None
    hf_hub_url: Optional[str] = None
    ms_hub_url: Optional[str] = None
    script_url: Optional[str] = None
    filename: Optional[str] = None
    formatting: Optional[str] = None
    ranking: bool
    subset: Optional[str] = None
    split: Optional[str] = None
    folder: Optional[str] = None
    columns: Optional[dict] = None
    status: DatasetStatusEnum
    num_samples: int
    created_at: datetime
    modified_at: datetime


class DatasetPaginatedResponse(PaginatedSuccessResponse):
    datasets: list[DatasetResponse]


class DatasetFilter(BaseModel):
    name: Optional[str] = None
    tags: Optional[str] = None
