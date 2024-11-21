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
from budapp.commons.exceptions import DatabaseException
from budapp.endpoint_ops.models import Endpoint
from budapp.model_ops.models import CloudModel, Model, PaperPublished
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


class PaperPublishedDataManager(DataManagerUtils):
    """Data manager for the PaperPublished model."""

    # async def get_paper_by_id(self, paper_id: UUID) -> Optional[PaperPublished]:
    #     """Retrieve a cloud model by its ID."""
    #     stmt = select(PaperPublished).where(PaperPublished.id == paper_id)
    #     return self.scalar_one_or_none(stmt)

    async def update_paper_by_id(self, paper_published: PaperPublished, update_data: Dict[str, Any]) -> PaperPublished:
        """Update specific fields of a cloud model and save using update_one."""
        # Update only the specified fields
        for field, value in update_data.items():
            if hasattr(paper_published, field):
                setattr(paper_published, field, value)

        # Use the update_one method to commit and refresh
        try:
            return self.update_one(paper_published)
        except DatabaseException as e:
            logger.error(f"Failed to update paper by id: {e}")
            raise


class ModelDataManager(DataManagerUtils):
    """Data manager for the Model model."""

    async def search_tags_by_name(
        self,
        search_value: str = "",
        offset: int = 0,
        limit: int = 10,
    ) -> Tuple[List[dict], int]:
        """Search tags by name with pagination, or fetch all tags if no search value is provided."""
        subquery = (
            select(func.jsonb_array_elements(Model.tags).label("tag"))
            .where(Model.is_active == True)
            .where(Model.tags.isnot(None))
        ).subquery()

        # Build the final query
        final_query = (
            select(
                func.jsonb_extract_path_text(subquery.c.tag, "name").label("name"),
                func.min(func.jsonb_extract_path_text(subquery.c.tag, "color")).label("color"),
            )
            .group_by("name")
            .order_by("name")
            .offset(offset)
            .limit(limit)
        )

        # Add the WHERE clause only if a search_value is provided
        if search_value:
            final_query = final_query.where(
                func.jsonb_extract_path_text(subquery.c.tag, "name").ilike(f"{search_value}%")
            )

        # Execute the query
        results = self.session.execute(final_query).all()
        tags = [{"name": res.name, "color": res.color} for res in results] if results else []

        # Total count query, adjusted to conditionally apply the search filter
        total_query = select(func.count(func.distinct(func.jsonb_extract_path_text(subquery.c.tag, "name"))))
        if search_value:
            total_query = total_query.where(
                func.jsonb_extract_path_text(subquery.c.tag, "name").ilike(f"{search_value}%")
            )
        total_count = self.session.execute(total_query).scalar()

        return tags, total_count

    async def search_tasks_by_name(
        self,
        search_value: str = "",
        offset: int = 0,
        limit: int = 10,
    ) -> Tuple[List[dict], int]:
        """Search tasks by name with pagination, or fetch all tags if no search value is provided."""
        subquery = (
            select(func.jsonb_array_elements(Model.tasks).label("task"))
            .where(Model.is_active)
            .where(Model.tasks.isnot(None))
        ).subquery()

        # Build the final query
        final_query = (
            select(
                func.jsonb_extract_path_text(subquery.c.task, "name").label("name"),
                func.min(func.jsonb_extract_path_text(subquery.c.task, "color")).label("color"),
            )
            .group_by("name")
            .order_by("name")
            .offset(offset)
            .limit(limit)
        )

        # Add the WHERE clause only if a search_value is provided
        if search_value:
            final_query = final_query.where(
                func.jsonb_extract_path_text(subquery.c.task, "name").ilike(f"{search_value}%")
            )

        # Execute the query
        results = self.session.execute(final_query).all()
        tasks = [{"name": res.name, "color": res.color} for res in results] if results else []

        # Total count query, adjusted to conditionally apply the search filter
        total_query = select(func.count(func.distinct(func.jsonb_extract_path_text(subquery.c.task, "name"))))
        if search_value:
            total_query = total_query.where(
                func.jsonb_extract_path_text(subquery.c.task, "name").ilike(f"{search_value}%")
            )
        total_count = self.session.execute(total_query).scalar()

        return tasks, total_count

    async def get_all_models(
        self,
        offset: int,
        limit: int,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[Model], int]:
        """List all models in the database."""
        await self.validate_fields(Model, filters)

        # Generate statements according to search or filters
        if search:
            search_conditions = await self.generate_search_stmt(Model, filters)
            stmt = (
                select(
                    Model,
                    func.count(Endpoint.id).filter(Endpoint.is_active == True).label("endpoints_count"),
                )
                .select_from(Model)
                .filter(or_(*search_conditions))
                .filter(Model.is_active == True)
                .outerjoin(Endpoint, Endpoint.model_id == Model.id)
                .group_by(Model.id)
            )
            count_stmt = (
                select(func.count()).select_from(Model).filter(or_(*search_conditions)).filter(Model.is_active == True)
            )
        else:
            stmt = (
                select(
                    Model,
                    func.count(Endpoint.id).filter(Endpoint.is_active == True).label("endpoints_count"),
                )
                .select_from(Model)
                .filter_by(**filters)
                .filter(Model.is_active == True)
                .outerjoin(Endpoint, Endpoint.model_id == Model.id)
                .group_by(Model.id)
            )
            count_stmt = select(func.count()).select_from(Model).filter_by(**filters).filter(Model.is_active == True)

        # Calculate count before applying limit and offset
        count = self.execute_scalar(count_stmt)

        # Apply limit and offset
        stmt = stmt.limit(limit).offset(offset)

        # Apply sorting
        if order_by:
            sort_conditions = await self.generate_sorting_stmt(Model, order_by)
            stmt = stmt.order_by(*sort_conditions)

        result = self.execute_all(stmt)

        return result, count


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
        # Tags and tasks are not filterable
        # Also remove from filters dict
        explicit_conditions = []
        json_filters = {"tags": filters.pop("tags", []), "tasks": filters.pop("tasks", [])}
        explicit_filters = {
            "modality": filters.pop("modality", []),
            "author": filters.pop("author", []),
            "model_size_min": filters.pop("model_size_min", None),
            "model_size_max": filters.pop("model_size_max", None),
        }

        # Validate the remaining filters
        await self.validate_fields(CloudModel, filters)

        if json_filters["tags"]:
            # Either TagA or TagB exist in tag field
            tag_conditions = or_(
                *[CloudModel.tags.cast(JSONB).contains([{"name": tag_name}]) for tag_name in json_filters["tags"]]
            )
            explicit_conditions.append(tag_conditions)

        if json_filters["tasks"]:
            # Either TaskA or TaskB exist in task field
            task_conditions = or_(
                *[CloudModel.tasks.cast(JSONB).contains([{"name": task_name}]) for task_name in json_filters["tasks"]]
            )
            explicit_conditions.append(task_conditions)

        if explicit_filters["modality"]:
            # Check any of modality present in the field
            modality_condition = CloudModel.modality.in_(explicit_filters["modality"])
            explicit_conditions.append(modality_condition)

        if explicit_filters["author"]:
            # Check any of author present in the field
            author_condition = CloudModel.modality.in_(explicit_filters["author"])
            explicit_conditions.append(author_condition)

        if explicit_filters["model_size_min"] is not None or explicit_filters["model_size_max"] is not None:
            # Add model size range condition
            size_conditions = []
            if explicit_filters["model_size_min"] is not None:
                size_conditions.append(CloudModel.model_size >= explicit_filters["model_size_min"])
            if explicit_filters["model_size_max"] is not None:
                size_conditions.append(CloudModel.model_size <= explicit_filters["model_size_max"])
            size_condition = and_(*size_conditions)
            explicit_conditions.append(size_condition)

        # Generate statements according to search or filters
        if search:
            search_conditions = await self.generate_search_stmt(CloudModel, filters)
            stmt = select(CloudModel).filter(or_(*search_conditions)).where(or_(*explicit_conditions))
            count_stmt = (
                select(func.count())
                .select_from(CloudModel)
                .filter(or_(*search_conditions))
                .where(or_(*explicit_conditions))
            )
        else:
            stmt = select(CloudModel).filter_by(**filters).where(and_(*explicit_conditions))
            count_stmt = (
                select(func.count()).select_from(CloudModel).filter_by(**filters).where(and_(*explicit_conditions))
            )

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
