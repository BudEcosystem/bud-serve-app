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

"""Contains dependency injection functions and utilities for the microservices, enabling modular and reusable components across the application."""

from collections.abc import AsyncGenerator

from sqlalchemy.orm import Session

from budapp.commons.database import SessionLocal


async def get_session() -> AsyncGenerator[Session, None]:
    """Create and yield an Session for database operations.

    This function is a dependency that provides an Session for use in FastAPI
    route handlers. It ensures that the session is properly closed after use.

    Yields:
        Session: An asynchronous SQLAlchemy session.

    Raises:
        SQLAlchemyError: If there's an error creating or using the session.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
