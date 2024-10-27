from typing import Dict, List, Tuple

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin
from budapp.model_ops.models import Provider as ProviderModel

from .crud import ProviderDataManager

logger = logging.get_logger(__name__)


class ProviderService(SessionMixin):
    """Provider service"""

    async def get_all_providers(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[ProviderModel], int]:
        """Get all providers."""

        return await ProviderDataManager(self.session).get_all_providers(offset, limit, filters, order_by, search)
