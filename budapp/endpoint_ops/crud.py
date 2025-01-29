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

"""The crud package, containing essential business logic, services, and routing configurations for the endpoint ops."""

from typing import Any, Dict, List, Tuple
from uuid import UUID

from sqlalchemy import and_, asc, desc, distinct, func, or_, select

from budapp.cluster_ops.models import Cluster as ClusterModel
from budapp.commons import logging
from budapp.commons.constants import EndpointStatusEnum
from budapp.commons.db_utils import DataManagerUtils
from budapp.model_ops.models import Model as Model

from ..project_ops.models import Project as ProjectModel
from .models import Endpoint as EndpointModel


logger = logging.get_logger(__name__)


class EndpointDataManager(DataManagerUtils):
    """Data manager for the Endpoint model."""

    async def get_all_active_endpoints(
        self,
        project_id: UUID,
        offset: int = 0,
        limit: int = 10,
        filters: Dict[str, Any] = {},
        order_by: List[Tuple[str, str]] = [],
        search: bool = False,
    ) -> Tuple[List[EndpointModel], int]:
        """Get all active endpoints from the database."""
        await self.validate_fields(EndpointModel, filters)

        # explicit conditions for order by model_name, cluster_name, modality
        explicit_conditions = []
        for field in order_by:
            if field[0] == "model_name":
                sorting_stmt = await self.generate_sorting_stmt(
                    Model,
                    [
                        ("name", field[1]),
                    ],
                )
                explicit_conditions.append(sorting_stmt[0])
            elif field[0] == "cluster_name":
                sorting_stmt = await self.generate_sorting_stmt(
                    ClusterModel,
                    [
                        ("name", field[1]),
                    ],
                )
                explicit_conditions.append(sorting_stmt[0])
            elif field[0] == "modality":
                sorting_stmt = await self.generate_sorting_stmt(
                    Model,
                    [
                        ("modality", field[1]),
                    ],
                )
                explicit_conditions.append(sorting_stmt[0])

        # Generate statements according to search or filters
        if search:
            search_conditions = await self.generate_search_stmt(EndpointModel, filters)
            stmt = (
                select(EndpointModel)
                .join(Model)
                .join(ClusterModel)
                .filter(or_(*search_conditions))
                .filter(
                    and_(EndpointModel.status != EndpointStatusEnum.DELETED, EndpointModel.project_id == project_id)
                )
            )
            count_stmt = (
                select(func.count())
                .select_from(EndpointModel)
                .join(Model)
                .join(ClusterModel)
                .filter(and_(*search_conditions))
                .filter(
                    and_(EndpointModel.status != EndpointStatusEnum.DELETED, EndpointModel.project_id == project_id)
                )
            )
        else:
            stmt = select(EndpointModel).join(Model).join(ClusterModel)
            count_stmt = select(func.count()).select_from(EndpointModel).join(Model).join(ClusterModel)
            for key, value in filters.items():
                stmt = stmt.filter(getattr(EndpointModel, key) == value)
                count_stmt = count_stmt.filter(getattr(EndpointModel, key) == value)
            stmt = stmt.filter(
                and_(EndpointModel.status != EndpointStatusEnum.DELETED, EndpointModel.project_id == project_id)
            )
            count_stmt = count_stmt.filter(
                and_(EndpointModel.status != EndpointStatusEnum.DELETED, EndpointModel.project_id == project_id)
            )

        # Calculate count before applying limit and offset
        count = self.execute_scalar(count_stmt)

        # Apply limit and offset
        stmt = stmt.limit(limit).offset(offset)

        # Apply sorting
        if order_by:
            sort_conditions = await self.generate_sorting_stmt(EndpointModel, order_by)
            # Extend sort conditions with explicit conditions
            sort_conditions.extend(explicit_conditions)
            stmt = stmt.order_by(*sort_conditions)

        result = self.scalars_all(stmt)

        return result, count

    async def get_all_endpoints_in_cluster(
        self, cluster_id: UUID, offset: int, limit: int, filters: Dict[str, Any], order_by: List[str], search: bool
    ) -> Tuple[List[EndpointModel], int, int, int]:
        """Get all endpoints in a cluster."""
        await self.validate_fields(EndpointModel, filters)

        # Base conditions
        base_conditions = [
            EndpointModel.cluster_id == cluster_id,
            EndpointModel.status != EndpointStatusEnum.DELETED,
        ]

        if search:
            search_conditions = await self.generate_search_stmt(EndpointModel, filters)

            stmt = (
                select(
                    EndpointModel,
                    ProjectModel.name.label("project_name"),
                    Model.name.label("model_name"),
                    func.sum(EndpointModel.total_replicas).label("total_workers"),
                )
                .join(ProjectModel, ProjectModel.id == EndpointModel.project_id)
                .join(Model, Model.id == EndpointModel.model_id)
                .filter(*base_conditions)
                .filter(and_(*search_conditions))
                .group_by(EndpointModel.id, ProjectModel.name, Model.name)
            )

            count_stmt = (
                select(func.count(distinct(EndpointModel.id)))
                .select_from(EndpointModel)
                .filter(*base_conditions)
                .filter(and_(*search_conditions))
            )
        else:
            filter_conditions = [getattr(EndpointModel, field) == value for field, value in filters.items()]
            stmt = (
                select(
                    EndpointModel,
                    ProjectModel.name.label("project_name"),
                    Model.name.label("model_name"),
                    func.sum(EndpointModel.total_replicas).label("total_workers"),
                )
                .join(ProjectModel, ProjectModel.id == EndpointModel.project_id)
                .join(Model, Model.id == EndpointModel.model_id)
                .filter(*base_conditions)
                .filter(*filter_conditions)
                .group_by(EndpointModel.id, ProjectModel.name, Model.name)
            )

            count_stmt = (
                select(func.count(distinct(EndpointModel.id)))
                .select_from(EndpointModel)
                .filter(*base_conditions)
                .filter(*filter_conditions)
            )

        # Get count
        count = self.execute_scalar(count_stmt)

        # Apply sorting and limit/offset
        stmt = stmt.limit(limit).offset(offset)

        if order_by:
            sort_conditions = await self.generate_sorting_stmt(EndpointModel, order_by)

            # Handle sorting for project_name, model_name, and total_workers
            for field, direction in order_by:
                sort_func = asc if direction == "asc" else desc
                if field == "project_name":
                    stmt = stmt.order_by(sort_func("project_name"))
                elif field == "model_name":
                    stmt = stmt.order_by(sort_func("model_name"))

            stmt = stmt.order_by(*sort_conditions)

        result = self.session.execute(stmt)

        return result, count

    async def get_cluster_count_details(self, cluster_id: UUID) -> Tuple[int, int, int, int]:
        """
        Retrieve cluster statistics including:
        - Total endpoints count (excluding deleted ones)
        - Running endpoints count
        - Sum of active replicas (workers)
        - Sum of total replicas (workers)

        Args:
            cluster_id (UUID): The ID of the cluster.

        Returns:
            Tuple[int, int, int, int]:
            (total_endpoints_count, running_endpoints_count, active_workers_count, total_workers_count)
        """
        query = select(
            func.count().filter(EndpointModel.status != EndpointStatusEnum.DELETED).label("total_endpoints"),
            func.count().filter(EndpointModel.status == EndpointStatusEnum.RUNNING).label("running_endpoints"),
            func.coalesce(
                func.sum(EndpointModel.active_replicas).filter(EndpointModel.status != EndpointStatusEnum.DELETED), 0
            ).label("active_workers"),
            func.coalesce(
                func.sum(EndpointModel.total_replicas).filter(EndpointModel.status != EndpointStatusEnum.DELETED), 0
            ).label("total_workers"),
        ).where(EndpointModel.cluster_id == cluster_id)

        result = self.session.execute(query)
        total_endpoints, running_endpoints, active_replicas, total_replicas = result.fetchone()

        return total_endpoints, running_endpoints, active_replicas, total_replicas
