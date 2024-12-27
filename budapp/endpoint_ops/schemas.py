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


"""Contains core Pydantic schemas used for data validation and serialization within the core services."""

from datetime import datetime

from pydantic import UUID4, BaseModel, ConfigDict

from budapp.cluster_ops.schemas import ClusterResponse
from budapp.commons.constants import EndpointStatusEnum
from budapp.commons.schemas import PaginatedSuccessResponse, SuccessResponse
from budapp.model_ops.schemas import ModelResponse


# Endpoint schemas


class EndpointCreate(BaseModel):
    """Create endpoint schema."""

    project_id: UUID4
    model_id: UUID4
    cluster_id: UUID4
    bud_cluster_id: UUID4
    name: str
    url: str
    namespace: str
    status: EndpointStatusEnum
    created_by: UUID4
    status_sync_at: datetime
    credential_id: UUID4 | None


class EndpointFilter(BaseModel):
    """Filter endpoint schema for filtering endpoints based on specific criteria."""

    name: str | None = None
    status: EndpointStatusEnum | None = None


class EndpointListResponse(BaseModel):
    """Endpoint list response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID4
    name: str
    status: EndpointStatusEnum
    model: ModelResponse
    cluster: ClusterResponse
    created_at: datetime
    modified_at: datetime


class EndpointPaginatedResponse(PaginatedSuccessResponse):
    """Endpoint paginated response schema."""

    endpoints: list[EndpointListResponse] = []


class EndpointCountResponse(SuccessResponse):
    """Endpoint count response schema."""

    total_endpoints_count: int
    running_endpoints_count: int
