from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import JSONB, DateTime, Enum, ForeignKey, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..commons.constants import CredentialTypeEnum
from ..commons.database import Base


class ProprietaryCredential(Base):
    """Proprietary model creds at global level : Credential model."""

    __tablename__ = "proprietary_credential"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = mapped_column(
        Enum(
            CredentialTypeEnum,
            name="proprietary_credential_type_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )

    # placeholder for api base, project, organization, etc.
    other_provider_creds: Mapped[dict] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(UTC))
    modified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now(UTC), onupdate=datetime.now(UTC))

    endpoints: Mapped[list["Endpoint"]] = relationship("Endpoint", back_populates="credential")
