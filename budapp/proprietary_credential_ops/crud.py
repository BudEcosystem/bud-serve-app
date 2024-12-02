from typing import Dict, List, Optional

from fastapi import status
from fastapi.exceptions import HTTPException
from sqlalchemy import and_, delete, func, select

from ..commons.db_utils import DataManagerUtils
from ..commons.logging import get_logger
from .models import ProprietaryCredential as ProprietaryCredentialModel


logger = get_logger(__name__)


class ProprietaryCredentialDataManager(DataManagerUtils):
    """Proprietary credential data manager class responsible for operations over database."""

    async def create_credential(self, credential: ProprietaryCredentialModel) -> ProprietaryCredentialModel:
        """Create a new credential in the database."""
        return await self.add_one(credential)

    async def retrieve_credential_by_fields(self, fields: Dict, missing_ok: bool = False) -> Optional[ProprietaryCredentialModel]:
        """Retrieve credential by fields."""
        await self.validate_fields(ProprietaryCredentialModel, fields)

        stmt = select(ProprietaryCredentialModel).filter_by(**fields)
        db_credential = await self.get_one_or_none(stmt)

        if not missing_ok and db_credential is None:
            logger.info("Credential not found in database")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Credential not found")

        return db_credential if db_credential else None

    async def get_credentials(self, fields: Dict) -> List[ProprietaryCredentialModel]:
        """List all proprietary credentials in the database."""
        await self.validate_fields(ProprietaryCredentialModel, fields)

        stmt = select(ProprietaryCredentialModel).filter_by(**fields)
        return await self.get_all(stmt)

    async def get_all_credentials(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Optional[Dict] = None,
        order_by: Optional[List] = None,
        search: bool = False,
    ) -> tuple[list[ProprietaryCredentialModel], int]:
        """List all credentials in the database."""
        filters = filters or {}
        order_by = order_by or []

        await self.validate_fields(ProprietaryCredentialModel, filters)

        # Generate statements according to search or filters
        if search:
            search_conditions = await self.generate_search_stmt(ProprietaryCredentialModel, filters)
            stmt = select(ProprietaryCredentialModel).filter_by(and_(*search_conditions))
            count_stmt = select(func.count()).select_from(ProprietaryCredentialModel).filter(and_(*search_conditions))
        else:
            stmt = select(ProprietaryCredentialModel).filter_by(**filters)
            count_stmt = select(func.count()).select_from(ProprietaryCredentialModel).filter_by(**filters)

        # Calculate count before applying limit and offset
        count = await self.execute_scalar_stmt(count_stmt)

        # Apply limit and offset
        stmt = stmt.limit(limit).offset(offset)

        # Apply sorting
        if order_by:
            sort_conditions = await self.generate_sorting_stmt(ProprietaryCredentialModel, order_by)
            stmt = stmt.order_by(*sort_conditions)

        result = await self.get_all(stmt)

        return result, count

    async def update_credential_by_fields(self, db_credential: ProprietaryCredentialModel, fields: Dict) -> ProprietaryCredentialModel:
        """Update a credential in the database."""
        await self.validate_fields(ProprietaryCredentialModel, fields)

        for field, value in fields.items():
            setattr(db_credential, field, value)

        return await self.update_one(db_credential)

    async def delete_credential(self, db_credential: ProprietaryCredentialModel):
        """Delete a credential from the database."""
        await self.delete_one(db_credential)

    async def delete_credential_by_fields(self, fields: Dict):
        """Delete credentials by fields."""
        await self.validate_fields(ProprietaryCredentialModel, fields)

        if len(fields) == 0:
            # raise error if fields is empty, Otherwise it will delete all entries
            raise ValueError("fields cannot be empty")

        stmt = delete(ProprietaryCredentialModel).filter_by(**fields)

        await self.execute_commit(stmt)
