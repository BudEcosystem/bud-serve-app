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

"""The crud package, containing essential business logic, services, and routing configurations for the cluster ops."""

from typing import Dict, List, Tuple
from uuid import UUID

from sqlalchemy import and_, func, select

from budapp.cluster_ops.models import Cluster
from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils

from ..commons.constants import ClusterStatusEnum, EndpointStatusEnum
from ..endpoint_ops.models import Endpoint


logger = logging.get_logger(__name__)


class ClusterDataManager(DataManagerUtils):
    """Data manager for the Cluster model."""

    async def get_all_clusters(
        self,
        offset: int,
        limit: int,
        filters: Dict = {},  # endpoint count need to consider in future
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[Cluster], int]:
        """List all clusters from the database."""
        await self.validate_fields(Cluster, filters)

        # Subquery to count endpoints per cluster
        endpoint_count_subquery = (
            select(
                Endpoint.cluster_id,
                func.count(Endpoint.id.distinct()).label("endpoint_count"),
            )
            .where(Endpoint.status != EndpointStatusEnum.DELETED)
            .group_by(Endpoint.cluster_id)
            .alias("endpoint_count_subquery")
        )

        # Generate statements based on search or filters
        if search:
            search_conditions = await self.generate_search_stmt(Cluster, filters)
            stmt = (
                select(
                    Cluster,
                    func.coalesce(endpoint_count_subquery.c.endpoint_count, 0).label("endpoint_count"),
                )
                .filter(and_(*search_conditions))
                .select_from(Cluster)
                .join(
                    endpoint_count_subquery,
                    Cluster.id == endpoint_count_subquery.c.cluster_id,
                    isouter=True,
                )
            )
            count_stmt = select(func.count()).select_from(Cluster).filter(and_(*search_conditions))
        else:
            stmt = (
                select(
                    Cluster,
                    func.coalesce(endpoint_count_subquery.c.endpoint_count, 0).label("endpoint_count"),
                )
                .filter_by(**filters)
                .select_from(Cluster)
                .join(
                    endpoint_count_subquery,
                    Cluster.id == endpoint_count_subquery.c.cluster_id,
                    isouter=True,
                )
            )
            count_stmt = select(func.count()).select_from(Cluster).filter_by(**filters)

        # Exclude deleted clusters
        stmt = stmt.filter(Cluster.status != ClusterStatusEnum.DELETED)
        count_stmt = count_stmt.filter(Cluster.status != ClusterStatusEnum.DELETED)

        # Calculate count before applying limit and offset
        count = self.execute_scalar(count_stmt)

        # Apply limit and offset
        stmt = stmt.limit(limit).offset(offset)

        # Apply sorting
        if order_by:
            sort_conditions = await self.generate_sorting_stmt(Cluster, order_by)
            stmt = stmt.order_by(*sort_conditions)

        result = self.execute_all(stmt)

        return result, count

    async def get_available_clusters_by_cluster_ids(self, cluster_ids: List[UUID]) -> List[Cluster]:
        """Get active clusters by cluster ids."""
        stmt = select(Cluster).filter(
            Cluster.cluster_id.in_(cluster_ids), Cluster.status == ClusterStatusEnum.AVAILABLE
        )
        count_stmt = (
            select(func.count())
            .select_from(Cluster)
            .filter(Cluster.cluster_id.in_(cluster_ids), Cluster.status == ClusterStatusEnum.AVAILABLE)
        )

        count = self.execute_scalar(count_stmt)
        result = self.scalars_all(stmt)

        return result, count
