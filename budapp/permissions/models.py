import json
from datetime import datetime
from typing import List, Union
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, String, Uuid
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship

from budapp.commons.database import Base
from budapp.user_ops.models import User


class Permission(Base):
    """Permission model."""

    __tablename__ = "permission"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    auth_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    scopes: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="permission")  # one-to-one

    @hybrid_property
    def scopes_list(self) -> Union[List[str], None]:
        """Get the scopes as a list of strings."""
        if not self.scopes:
            return []
        return json.loads(self.scopes)
