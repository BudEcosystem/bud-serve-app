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

from pydantic import UUID4, BaseModel

from budapp.commons.constants import EndpointStatusEnum


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
    replicas: int
    status: EndpointStatusEnum
    created_by: UUID4
    status_sync_at: datetime
