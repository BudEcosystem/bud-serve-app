from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, String, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from budapp.commons.constants import ModalityEnum, ModelProviderTypeEnum, ModelTypeEnum
from budapp.commons.database import Base


class Model(Base):
    """Model for a AI model."""

    __tablename__ = "model"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=True)
    tags: Mapped[list[dict]] = mapped_column(JSONB, nullable=True)
    tasks: Mapped[list[dict]] = mapped_column(JSONB, nullable=True)
    author: Mapped[str] = mapped_column(String, nullable=False)
    model_size: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    icon: Mapped[Optional[str]] = mapped_column(String, nullable=False)
    github_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    huggingface_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    website_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    modality: Mapped[str] = mapped_column(
        Enum(
            ModalityEnum,
            name="modality_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    type: Mapped[str] = mapped_column(
        Enum(
            ModelTypeEnum,
            name="model_type_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    provider_type: Mapped[str] = mapped_column(
        Enum(
            ModelProviderTypeEnum,
            name="model_provider_type_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    uri: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
