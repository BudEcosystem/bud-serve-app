from typing import List

from sqlalchemy import select

from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils
from budapp.model_ops.models import Provider


logger = logging.get_logger(__name__)


class ProviderDataManager(DataManagerUtils):
    """Data manager for the Provider model."""

    async def get_all_providers_by_type(self, provider_types: List[str]) -> List[Provider]:
        """Get all providers from the database."""
        stmt = select(Provider).filter(Provider.type.in_(provider_types))
        return self.scalars_all(stmt)
