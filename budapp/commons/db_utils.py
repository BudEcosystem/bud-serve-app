from typing import Any, Dict, Optional, Type

from commons.exceptions import DatabaseException
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.sql import Executable

from . import logging


logger = logging.get_logger(__name__)


class SessionMixin:
    """A mixin class that provides an instance of a database session.

    This mixin is designed to be used with classes that require database access.
    It initializes and stores a SQLAlchemy Session object for database operations.

    Attributes:
        session (Session): An instance of SQLAlchemy Session for database operations.

    Example:
        class MyDatabaseHandler(SessionMixin):
            def __init__(self, session):
                super().__init__(session)

            def some_db_operation(self):
                # Use self.session to perform database operations
                pass
    """

    def __init__(self, session: Session) -> None:
        """Initialize the SessionMixin with a database session.

        Args:
            session (Session): An instance of SQLAlchemy Session to be used for database operations.
        """
        self.session = session


class SQLAlchemyMixin(SessionMixin):
    """A mixin class that provides methods for database operations."""

    def add_one(self, model: object) -> object:
        """Add a single model instance to the database.

        This method adds the given model to the session, commits the transaction,
        and refreshes the model to ensure it reflects the current state in the database.

        Args:
            model (Any): The SQLAlchemy model instance to be added to the database.

        Returns:
            Any: The added and refreshed model instance.

        Raises:
            DatabaseException: If there's an error during the database operation.
        """
        try:
            self.session.add(model)
            self.session.commit()
            self.session.refresh(model)
            return model
        except (Exception, SQLAlchemyError) as e:
            self.session.rollback()
            logger.exception(f"Failed to add one model to database: {e}")
            raise DatabaseException("Unable to add model to database") from e

    def scalar_one_or_none(self, stmt: Executable) -> object:
        """Execute a SQL statement and return a single result or None.

        This method executes the given SQL statement and returns either a single
        scalar result or None if no results are found.

        Args:
            stmt (Executable): The SQLAlchemy statement to be executed.

        Returns:
            Any: The scalar result of the query, or None if no results are found.

        Raises:
            DatabaseException: If there's an error during the database operation.
        """
        try:
            return self.session.scalar_one_or_none(stmt)
        except (Exception, SQLAlchemyError) as e:
            logger.exception(f"Failed to get one model from database: {e}")
            raise DatabaseException("Unable to get model from database") from e

    def update_one(self, model: object) -> object:
        """Update a single model instance in the database.

        This method commits the current session (which should contain the updates to the model),
        and then refreshes the model to ensure it reflects the current state in the database.

        Args:
            model (Any): The SQLAlchemy model instance to be updated in the database.

        Returns:
            Any: The updated and refreshed model instance.

        Raises:
            DatabaseException: If there's an error during the database operation.
        """
        try:
            self.session.commit()
            self.session.refresh(model)
            return model
        except (Exception, SQLAlchemyError) as e:
            self.session.rollback()
            logger.exception(f"Failed to update one model in database: {e}")
            raise DatabaseException("Unable to update model in database") from e


class DataManagerUtils(SQLAlchemyMixin):
    """Utility class for data management operations."""

    @staticmethod
    async def validate_fields(model: Type[DeclarativeBase], fields: Dict[str, Any]) -> None:
        """Validate that the given fields exist in the SQLAlchemy model.

        Args:
            model (Type[DeclarativeBase]): The SQLAlchemy model class to validate against.
            fields (Dict[str, Any]): A dictionary of field names and their values to validate.

        Raises:
            DatabaseException: If an invalid field is found in the input.
        """
        for field in fields:
            if not hasattr(model, field):
                logger.error(f"Invalid field: '{field}' not found in {model.__name__} model")
                raise DatabaseException(f"Invalid field: '{field}' not found in {model.__name__} model")

    async def insert_one(self, model: object) -> object:
        """Insert a single model instance into the database.

        This method is an alias for the `add_one` method, providing a more
        intuitive name for the insertion operation.

        Args:
            model (object): The model instance to be inserted into the database.

        Returns:
            object: The inserted model instance, potentially with updated
                attributes (e.g., auto-generated ID).

        Raises:
            Any exceptions that may be raised by the underlying `add_one` method.
        """
        return self.add_one(model)

    async def retrieve_by_fields(
        self, model: Type[DeclarativeBase], fields: Dict[str, Any], missing_ok: bool = False
    ) -> Optional[DeclarativeBase]:
        """Retrieve a model instance from the database based on given fields.

        This method queries the database for a model instance matching the provided fields.
        If the instance is not found and missing_ok is False, it raises an HTTPException.

        Args:
            model (Type[DeclarativeBase]): The SQLAlchemy model class to query.
            fields (Dict): A dictionary of field names and their values to filter by.
            missing_ok (bool, optional): If True, return None when no instance is found
                                         instead of raising an exception. Defaults to False.

        Returns:
            Optional[DeclarativeBase]: The found model instance, or None if not found and missing_ok is True.

        Raises:
            HTTPException: If the model instance is not found and missing_ok is False.
            DatabaseException: If there's an error in field validation or database operation.
        """
        await self.validate_fields(model, fields)

        stmt = select(model).filter_by(**fields)
        db_model = self.scalar_one_or_none(stmt)

        if not missing_ok and db_model is None:
            logger.info(f"{model.__name__} not found in database")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{model.__name__} not found")

        return db_model if db_model else None

    async def update_by_fields(self, model: Type[DeclarativeBase], fields: Dict[str, Any]) -> object:
        """Update a model instance with the given fields.

        This method updates the attributes of the provided model instance with the values
        in the fields dictionary and then persists the changes to the database.

        Args:
            model (Type[DeclarativeBase]): The SQLAlchemy model instance to update.
            fields (Dict): A dictionary of field names and their new values.

        Returns:
            DeclarativeBase: The updated model instance.

        Raises:
            DatabaseException: If there's an error in field validation or database operation.
        """
        await self.validate_fields(model, fields)

        for field, value in fields.items():
            setattr(model, field, value)

        return self.update_one(model)
