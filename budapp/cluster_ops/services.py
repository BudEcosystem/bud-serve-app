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

"""The cluster ops services. Contains business logic for model ops."""

from typing import List, Tuple
from .schemas import Cluster
from .crud import ClusterDataManager

class ClusterService:
    """Cluster service."""

    async def get_all_clusters(self, offset: int = 0, limit: int = 10) -> Tuple[List[Cluster], int]:
        """Get all clusters (dummy implementation)."""
        return await ClusterDataManager().get_all_clusters(offset, limit)