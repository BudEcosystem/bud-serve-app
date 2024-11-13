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

from typing import List, Tuple, Dict

from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils

from sqlalchemy import and_, desc, func, or_, select

from budapp.cluster_ops.models import Cluster
from .schemas import ClusterResponse

logger = logging.get_logger(__name__)


class ClusterDataManager(DataManagerUtils):
    """Data manager for the Cluster model."""

    async def get_all_clusters(
        self,
        offset: int,
        limit: int,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[ClusterResponse], int]:
        """List all clusters from the database."""

        await self.validate_fields(Cluster, filters)

        # Generate statements based on search or filters
        if search:
            search_conditions = await self.generate_search_stmt(Cluster, filters)
            stmt = select(Cluster).filter(and_(*search_conditions))
            count_stmt = select(func.count()).select_from(Cluster).filter(and_(*search_conditions))
        else:
            stmt = select(Cluster).filter_by(**filters)
            count_stmt = select(func.count()).select_from(Cluster).filter_by(**filters)

        # Calculate count before applying limit and offset
        count = self.execute_scalar(count_stmt)

        # Apply limit and offset
        stmt = stmt.limit(limit).offset(offset)

        # Apply sorting
        if order_by:
            sort_conditions = await self.generate_sorting_stmt(Cluster, order_by)
            stmt = stmt.order_by(*sort_conditions)

        result = self.scalars_all(stmt)
        print("cluster result:", result)

        return result, count