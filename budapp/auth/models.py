from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Enum, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from budapp.commons.constants import TokenTypeEnum
from budapp.commons.database import Base


class Token(Base):
    """Token model."""

    __tablename__ = "token"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    auth_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    secret_key: Mapped[str] = mapped_column(String)
    token_hash: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(
        Enum(
            TokenTypeEnum,
            name="token_type_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    blacklisted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
