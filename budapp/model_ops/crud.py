from typing import Any, Dict, List, Tuple

from sqlalchemy import and_, func, or_, select

from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils
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
