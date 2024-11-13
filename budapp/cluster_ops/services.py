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

from typing import List, Tuple, Dict
from .schemas import ClusterResponse
from .crud import ClusterDataManager
from budapp.commons.db_utils import SessionMixin


class ClusterService(SessionMixin):
    """Cluster service."""

    async def get_all_active_clusters(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[ClusterResponse], int]:
        """Get all active clusters."""
        filters_dict = filters
        filters_dict["is_active"] = True

        clusters, count = await ClusterDataManager(self.session).get_all_clusters(
            offset, limit, filters_dict, order_by, search
        )
         # Add dummy data and additional fields
        updated_clusters = []
        for cluster in clusters:
            updated_cluster = ClusterResponse(**{
                "id": cluster.id,
                "cluster_id": cluster.cluster_id,
                "name": cluster.name,
                "icon": cluster.icon,
                "created_at": cluster.created_at,
                "modified_at": cluster.modified_at,
                "endpoint_count": 12,
                "status": cluster.status,
                "resources": {
                    "available_nodes": cluster.available_workers,
                    "total_nodes": cluster.total_workers,     
                    "gpu_count": 4,         
                    "cpu_count": 8,         
                    "hpu_count": 2,         
                }
            })
            updated_clusters.append(updated_cluster)

        return updated_clusters, count