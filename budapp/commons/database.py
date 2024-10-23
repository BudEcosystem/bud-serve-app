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

"""Provides utility functions for managing the database."""

from sqlalchemy import Engine, MetaData, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from budapp.commons.config import app_settings


def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine instance.

    This function initializes a new SQLAlchemy engine using the PostgreSQL
    connection URL specified in the application settings. The engine is
    configured with echo=True for SQL query logging.

    Returns:
        Engine: A SQLAlchemy Engine instance connected to the PostgreSQL database.

    Raises:
        SQLAlchemyError: If there's an error creating the engine or connecting to the database.
    """
    return create_engine(app_settings.postgres_url, echo=True)


# Create sqlalchemy engine
engine = get_engine()

# Create session class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Constraint naming convention to fix alembic autogenerate command issues
# https://docs.sqlalchemy.org/en/20/core/constraints.html#constraint-naming-conventions
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata_obj = MetaData(naming_convention=convention)

# Create base class for creating models
Base = declarative_base(metadata=metadata_obj)
