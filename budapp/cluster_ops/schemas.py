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


"""Contains core Pydantic schemas used for data validation and serialization within the cluster ops services."""

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
)
from typing import List, Dict
from datetime import datetime

from budapp.commons.schemas import PaginatedSuccessResponse


class Cluster(BaseModel):
    """Cluster schema."""

    id: UUID4
    name: str
    icon: str
    created_at: datetime
    endpoint_count: int
    status: str
    resources: Dict[str, int]

class ClusterResponse(PaginatedSuccessResponse):
    """Cluster response schema."""

    model_config = ConfigDict(extra="ignore")

    clusters: List[Cluster]
