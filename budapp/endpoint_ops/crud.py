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

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import and_, asc, case, cast, desc, distinct, func, literal, or_, select, update
from sqlalchemy.dialects.postgresql import JSONB

from budapp.cluster_ops.models import Cluster as ClusterModel
from budapp.commons import logging
from budapp.commons.constants import EndpointStatusEnum
from budapp.commons.db_utils import DataManagerUtils
from budapp.model_ops.models import CloudModel
from budapp.model_ops.models import Model as Model

from ..commons.constants import AdapterStatusEnum, ModelProviderTypeEnum
from ..commons.helpers import get_param_range
from ..project_ops.models import Project as ProjectModel
from .models import Adapter as AdapterModel
from .models import Endpoint as EndpointModel


logger = logging.get_logger(__name__)


class EndpointDataManager(DataManagerUtils):
    """Data manager for the Endpoint model."""

    async def get_missing_endpoints(self, endpoint_ids: List[UUID]):
        stmt = select(EndpointModel).where(
            and_(EndpointModel.id.in_(endpoint_ids), EndpointModel.status != EndpointStatusEnum.DELETED)
        )
        result = self.scalars_all(stmt)
        existing_ids = {ep.id for ep in result}
        missing_ids = set(endpoint_ids) - existing_ids
        return list(missing_ids)

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

    async def get_all_running_endpoints(self, project_id: UUID) -> List[EndpointModel]:
        """Get all running endpoints for a given project."""
        stmt = select(EndpointModel).filter(
            and_(
                EndpointModel.status != EndpointStatusEnum.DELETED,
                EndpointModel.status != EndpointStatusEnum.DELETING,
                EndpointModel.project_id == project_id,
            )
        )
        return self.scalars_all(stmt)

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
                    ProjectModel,
                    Model,
                    EndpointModel.total_replicas.label("total_workers"),
                    EndpointModel.active_replicas.label("active_workers"),
                )
                .join(ProjectModel, ProjectModel.id == EndpointModel.project_id)
                .join(Model, Model.id == EndpointModel.model_id)
                .filter(*base_conditions)
                .filter(and_(*search_conditions))
                .group_by(EndpointModel.id, ProjectModel.id, Model.id)
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
                    ProjectModel,
                    Model,
                    EndpointModel.total_replicas.label("total_workers"),
                    EndpointModel.active_replicas.label("active_workers"),
                )
                .join(ProjectModel, ProjectModel.id == EndpointModel.project_id)
                .join(Model, Model.id == EndpointModel.model_id)
                .filter(*base_conditions)
                .filter(*filter_conditions)
                .group_by(EndpointModel.id, ProjectModel.id, Model.id)
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
                    stmt = stmt.order_by(sort_func(ProjectModel.name))
                elif field == "model_name":
                    stmt = stmt.order_by(sort_func(Model.name))
                elif field == "total_workers":
                    stmt = stmt.order_by(sort_func("total_workers"))
                elif field == "active_workers":
                    stmt = stmt.order_by(sort_func("active_workers"))

            stmt = stmt.order_by(*sort_conditions)

        result = self.session.execute(stmt)

        return result, count

    async def get_cluster_count_details(self, cluster_id: UUID) -> Tuple[int, int, int, int]:
        """Retrieve cluster statistics including:
        - Total endpoints count (excluding deleted ones)
        - Running endpoints count
        - Sum of active replicas (workers)
        - Sum of total replicas (workers).

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

    async def get_all_playground_deployments(
        self,
        project_ids: List[UUID],
        offset: int,
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[List[Tuple[str, str]]] = None,
        search: bool = False,
    ) -> Tuple[List[EndpointModel], int]:
        """Get all playground deployments."""
        filters = filters or {}
        order_by = order_by or []

        # Define explicit conditions
        explicit_conditions = []
        json_filters = {"tags": filters.pop("tags", []), "tasks": filters.pop("tasks", [])}
        explicit_filters = {
            "modality": filters.pop("modality", []),
            "model_size": filters.pop("model_size", None),
            "model_name": filters.pop("model_name", None),
        }

        # Validate the remaining filters
        await self.validate_fields(EndpointModel, filters)

        if json_filters["tags"]:
            # Either TagA or TagB exist in tag field
            tag_conditions = or_(
                *[Model.tags.cast(JSONB).contains([{"name": tag_name}]) for tag_name in json_filters["tags"]]
            )
            explicit_conditions.append(tag_conditions)

        if json_filters["tasks"]:
            # Either TaskA or TaskB exist in task field
            task_conditions = or_(
                *[Model.tasks.cast(JSONB).contains([{"name": task_name}]) for task_name in json_filters["tasks"]]
            )
            explicit_conditions.append(task_conditions)

        if explicit_filters["modality"]:
            # Check any of modality present in the field
            modality_condition = Model.modality.in_(explicit_filters["modality"])
            explicit_conditions.append(modality_condition)

        if explicit_filters["model_size"]:
            # Convert model size to a pre-defined range of numbers
            model_size_min, model_size_max = get_param_range(explicit_filters["model_size"])
            explicit_conditions.append(Model.model_size.between(model_size_min, model_size_max))

        # Generate statements according to search or filters
        if search:
            if explicit_filters["model_name"]:
                # For search, query using like operator
                model_name_condition = Model.name.ilike(f"%{explicit_filters['model_name']}%")
                explicit_conditions.append(model_name_condition)

            search_conditions = await self.generate_search_stmt(EndpointModel, filters)
            stmt = (
                select(
                    EndpointModel,
                    case(
                        (Model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL, cast(CloudModel.input_cost, JSONB)),
                        else_=literal(None),
                    ).label("input_cost"),
                    case(
                        (
                            Model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL,
                            cast(CloudModel.output_cost, JSONB),
                        ),
                        else_=literal(None),
                    ).label("output_cost"),
                    case(
                        (Model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL, CloudModel.max_input_tokens),
                        (
                            Model.provider_type == ModelProviderTypeEnum.HUGGING_FACE,
                            Model.architecture_text_config["context_length"].as_integer(),
                        ),
                        else_=literal(None),
                    ).label("context_length"),
                )
                .join(Model, EndpointModel.model_id == Model.id)
                .outerjoin(
                    CloudModel,
                    and_(Model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL, Model.uri == CloudModel.uri),
                )
                .filter(or_(*search_conditions, *explicit_conditions))
                .filter(EndpointModel.status == EndpointStatusEnum.RUNNING)
                .filter(EndpointModel.project_id.in_(project_ids))
            )
            count_stmt = (
                select(func.count(distinct(EndpointModel.id)))
                .join(Model, EndpointModel.model_id == Model.id)
                .filter(or_(*search_conditions, *explicit_conditions))
                .filter(EndpointModel.status == EndpointStatusEnum.RUNNING)
                .filter(EndpointModel.project_id.in_(project_ids))
            )
        else:
            if explicit_filters["model_name"]:
                model_name_condition = Model.name == explicit_filters["model_name"]
                explicit_conditions.append(model_name_condition)

            stmt = (
                select(
                    EndpointModel,
                    case(
                        (Model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL, cast(CloudModel.input_cost, JSONB)),
                        else_=literal(None),
                    ).label("input_cost"),
                    case(
                        (
                            Model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL,
                            cast(CloudModel.output_cost, JSONB),
                        ),
                        else_=literal(None),
                    ).label("output_cost"),
                    case(
                        (Model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL, CloudModel.max_input_tokens),
                        (
                            Model.provider_type == ModelProviderTypeEnum.HUGGING_FACE,
                            Model.architecture_text_config["context_length"].as_integer(),
                        ),
                        else_=literal(None),
                    ).label("context_length"),
                )
                .filter_by(**filters)
                .join(Model, EndpointModel.model_id == Model.id)
                .outerjoin(
                    CloudModel,
                    and_(Model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL, Model.uri == CloudModel.uri),
                )
                .where(and_(*explicit_conditions))
                .filter(EndpointModel.status == EndpointStatusEnum.RUNNING)
                .filter(EndpointModel.project_id.in_(project_ids))
            )
            count_stmt = (
                select(func.count(distinct(EndpointModel.id)))
                .filter_by(**filters)
                .join(Model, EndpointModel.model_id == Model.id)
                .where(and_(*explicit_conditions))
                .filter(EndpointModel.status == EndpointStatusEnum.RUNNING)
                .filter(EndpointModel.project_id.in_(project_ids))
            )

        # Calculate count before applying limit and offset
        count = self.execute_scalar(count_stmt)

        # Apply limit and offset
        stmt = stmt.limit(limit).offset(offset)

        # Apply sorting
        if order_by:
            sort_conditions = await self.generate_sorting_stmt(EndpointModel, order_by)
            stmt = stmt.order_by(*sort_conditions)

        result = self.execute_all(stmt)

        return result, count

    async def mark_as_deprecated(self, model_ids: List[UUID]) -> None:
        """Mark models as deprecated.

        Args:
            model_ids (List[UUID]): List of model ids to mark as deprecated.
        """
        stmt = update(EndpointModel).where(EndpointModel.model_id.in_(model_ids)).values(is_deprecated=True)
        self.session.execute(stmt)
        self.session.commit()


class AdapterDataManager(DataManagerUtils):
    """Data manager for the Adapter model."""

    async def get_all_active_adapters(
        self,
        endpoint_id: UUID,
        offset: int = 0,
        limit: int = 10,
        filters: Dict[str, Any] = {},
        order_by: List[Tuple[str, str]] = [],
        search: bool = False,
    ) -> List[AdapterModel]:
        """Get all active adapters for a given endpoint.

        Args:
            endpoint_id (UUID): The ID of the endpoint.
            offset (int, optional): The offset for pagination. Defaults to 0.
            limit (int, optional): The limit for pagination. Defaults to 10.
            filters (Dict[str, Any], optional): Filters to apply. Defaults to {}.
            order_by (List[Tuple[str, str]], optional): The order by conditions. Defaults to [].
            search (bool, optional): Whether to perform a search. Defaults to False.

        Returns:
            List[AdapterModel]: A list of active adapters.
        """
        await self.validate_fields(AdapterModel, filters)

        # Generate statements according to search or filters
        if search:
            search_conditions = await self.generate_search_stmt(AdapterModel, filters)
            stmt = (
                select(AdapterModel)
                .join(EndpointModel)
                .join(Model)
                .filter(or_(*search_conditions))
                .filter(
                    and_(AdapterModel.endpoint_id == endpoint_id, AdapterModel.status != AdapterStatusEnum.DELETED)
                )
            )

            count_stmt = (
                select(func.count())
                .select_from(AdapterModel)
                .join(EndpointModel)
                .join(Model)
                .filter(or_(*search_conditions))
                .filter(
                    and_(AdapterModel.endpoint_id == endpoint_id, AdapterModel.status != AdapterStatusEnum.DELETED)
                )
            )
        else:
            stmt = select(AdapterModel).join(EndpointModel).join(Model)
            count_stmt = select(func.count()).select_from(AdapterModel).join(EndpointModel).join(Model)
            for key, value in filters.items():
                stmt = stmt.filter(getattr(AdapterModel, key) == value)
                count_stmt = count_stmt.filter(getattr(AdapterModel, key) == value)
            stmt = stmt.filter(
                and_(AdapterModel.endpoint_id == endpoint_id, AdapterModel.status != AdapterStatusEnum.DELETED)
            )
            count_stmt = count_stmt.filter(
                and_(AdapterModel.endpoint_id == endpoint_id, AdapterModel.status != AdapterStatusEnum.DELETED)
            )
        # Calculate count before applying limit and offset
        count = self.execute_scalar(count_stmt)

        # Apply limit and offset
        stmt = stmt.limit(limit).offset(offset)

        # Apply sorting
        if order_by:
            sort_conditions = await self.generate_sorting_stmt(AdapterModel, order_by)
            stmt = stmt.order_by(*sort_conditions)

        result = self.scalars_all(stmt)
        logger.info("all adapters result: %s", result)
        return result, count

    async def get_all_adapters_in_project(self, project_id: UUID) -> Tuple[List[AdapterModel], int]:
        """Get all adapters in a project."""
        stmt = select(AdapterModel).join(EndpointModel).filter(EndpointModel.project_id == project_id)
        count_stmt = (
            select(func.count())
            .select_from(AdapterModel)
            .join(EndpointModel)
            .filter(EndpointModel.project_id == project_id)
        )
        count = self.execute_scalar(count_stmt)
        result = self.scalars_all(stmt)
        logger.info("all adapters result: %s", result)
        return result, count
