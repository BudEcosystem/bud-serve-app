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

from typing import Any, Dict, List, Tuple

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.dialects.postgresql import JSONB

from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils
from budapp.model_ops.models import CloudModel
from budapp.model_ops.models import Provider as ProviderModel


logger = logging.get_logger(__name__)


class ProviderDataManager(DataManagerUtils):
    """Data manager for the Provider model."""

    async def get_all_providers_by_type(self, provider_types: List[str]) -> List[ProviderModel]:
        """Get all providers from the database."""
        stmt = select(ProviderModel).filter(ProviderModel.type.in_(provider_types))
        return self.scalars_all(stmt)

    async def get_all_providers(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict[str, Any] = {},
        order_by: List[Tuple[str, str]] = [],
        search: bool = False,
    ) -> Tuple[List[ProviderModel], int]:
        """Get all providers from the database."""
        await self.validate_fields(ProviderModel, filters)

        # Generate statements according to search or filters
        if search:
            search_conditions = await self.generate_search_stmt(ProviderModel, filters)
            stmt = select(ProviderModel).filter(or_(*search_conditions))
            count_stmt = select(func.count()).select_from(ProviderModel).filter(and_(*search_conditions))
        else:
            stmt = select(ProviderModel).filter_by(**filters)
            count_stmt = select(func.count()).select_from(ProviderModel).filter_by(**filters)

        # Calculate count before applying limit and offset
        count = self.execute_scalar(count_stmt)

        # Apply limit and offset
        stmt = stmt.limit(limit).offset(offset)

        # Apply sorting
        if order_by:
            sort_conditions = await self.generate_sorting_stmt(ProviderModel, order_by)
            stmt = stmt.order_by(*sort_conditions)

        result = self.scalars_all(stmt)

        return result, count


class ModelDataManager(DataManagerUtils):
    """Data manager for the Model model."""

    pass


class CloudModelDataManager(DataManagerUtils):
    """Data manager for the CloudModel model."""

    async def get_all_cloud_models_by_source_uris(self, provider: str, uris: List[str]) -> List[CloudModel]:
        """Get all cloud models from the database."""
        stmt = select(CloudModel).filter(CloudModel.uri.in_(uris), CloudModel.source == provider)
        return self.scalars_all(stmt)

    async def get_all_cloud_models(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict[str, Any] = {},
        order_by: List[Tuple[str, str]] = [],
        search: bool = False,
    ) -> Tuple[List[CloudModel], int]:
        """Get all cloud models from the database."""
        await self.validate_fields(CloudModel, filters)

        # Tags and tasks are not filterable
        json_filters = {"tags": [], "tasks": []}
        if "tags" in filters:
            json_filters["tags"] = filters["tags"]
            del filters["tags"]
        if "tasks" in filters:
            json_filters["tasks"] = filters["tasks"]
            del filters["tasks"]

        conditions = [CloudModel.tags.cast(JSONB).contains([{"name": tag_name}]) for tag_name in json_filters["tags"]]

        conditions.extend(
            [CloudModel.tasks.cast(JSONB).contains([{"name": task_name}]) for task_name in json_filters["tasks"]]
        )

        # Generate statements according to search or filters
        if search:
            search_conditions = await self.generate_search_stmt(CloudModel, filters)
            stmt = select(CloudModel).filter(or_(*search_conditions)).where(or_(*conditions))
            count_stmt = (
                select(func.count()).select_from(CloudModel).filter(or_(*search_conditions)).where(or_(*conditions))
            )
        else:
            stmt = select(CloudModel).filter_by(**filters).where(and_(*conditions))
            count_stmt = select(func.count()).select_from(CloudModel).filter_by(**filters).where(and_(*conditions))

        # Calculate count before applying limit and offset
        count = self.execute_scalar(count_stmt)

        # Apply limit and offset
        stmt = stmt.limit(limit).offset(offset)

        # Apply sorting
        if order_by:
            sort_conditions = await self.generate_sorting_stmt(CloudModel, order_by)
            stmt = stmt.order_by(*sort_conditions)

        result = self.scalars_all(stmt)

        return result, count

    async def get_all_recommended_tags(
        self,
        offset: int = 0,
        limit: int = 10,
    ) -> Tuple[List[CloudModel], int]:
        """Get all recommended tags from the database."""
        stmt = (
            (
                select(
                    func.jsonb_array_elements(CloudModel.tags).op("->>")("name").label("name"),
                    func.jsonb_array_elements(CloudModel.tags).op("->>")("color").label("color"),
                    func.count().label("count"),
                )
                .select_from(CloudModel)
                .where(CloudModel.tags.is_not(None))
                .group_by(
                    func.jsonb_array_elements(CloudModel.tags).op("->>")("name"),
                    func.jsonb_array_elements(CloudModel.tags).op("->>")("color"),
                )
            )
            .union_all(
                select(
                    func.jsonb_array_elements(CloudModel.tasks).op("->>")("name").label("name"),
                    func.jsonb_array_elements(CloudModel.tasks).op("->>")("color").label("color"),
                    func.count().label("count"),
                )
                .select_from(CloudModel)
                .where(CloudModel.tasks.is_not(None))
                .group_by(
                    func.jsonb_array_elements(CloudModel.tasks).op("->>")("name"),
                    func.jsonb_array_elements(CloudModel.tasks).op("->>")("color"),
                )
            )
            .order_by(desc("count"), "name")
            .offset(offset)
            .limit(limit)
        )

        count_stmt = select(func.count()).select_from(stmt)

        count = self.execute_scalar(count_stmt)

        result = self.execute_all(stmt)

        return result, count
