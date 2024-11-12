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

"""The crud package, containing essential business logic, services, and routing configurations for the model ops."""

from typing import List, Tuple

from uuid import UUID


from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils


logger = logging.get_logger(__name__)

from typing import List, Tuple
from .schemas import Cluster

class ClusterDataManager(DataManagerUtils):
    """Data manager for the Cluster model."""

    async def get_all_clusters(self, offset: int = 0, limit: int = 10) -> Tuple[List[Cluster], int]:
        """Get all clusters from dummy data."""
        dummy_clusters = [
            Cluster(
                id=UUID("123e4567-e89b-42d3-a456-426614174000"),  # Updated to a valid UUID4
                name="Cluster A",
                icon="icons/clusters/cluster1.png",
                created_at="2025-12-10T00:00:00Z",
                endpoint_count=12,
                status="Available",
                resources={
                    "available_nodes": 10,
                    "total_nodes": 50,
                    "gpu_count": 0,
                    "cpu_count": 0,
                    "hpu_count": 0,
                },
            ),
            Cluster(
                id=UUID("223e4567-e89b-42d3-a456-426614174000"),  # Updated to a valid UUID4
                name="Cluster B",
                icon="icons/clusters/cluster2.png",
                created_at="2025-12-11T00:00:00Z",
                endpoint_count=20,
                status="Available",
                resources={
                    "available_nodes": 15,
                    "total_nodes": 60,
                    "gpu_count": 2,
                    "cpu_count": 20,
                    "hpu_count": 5,
                },
            ),
        ]
        # Apply limit and offset on the dummy data
        total_count = len(dummy_clusters)
        clusters = dummy_clusters[offset : offset + limit]
        return clusters, total_count