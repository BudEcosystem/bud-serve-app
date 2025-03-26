from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin


logger = logging.get_logger(__name__)

class DatasetService(SessionMixin):
    """Dataset service."""

    async def get_datasets(self, offset, limit, filters_dict, order_by, search):
        """Get all datasets."""
        pass