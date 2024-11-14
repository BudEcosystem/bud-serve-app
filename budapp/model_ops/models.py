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

"""The model ops package, containing essential business logic, services, and routing configurations for the model ops."""

from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, ForeignKey, String, Uuid, ARRAY, Integer
from sqlalchemy.dialects.postgresql import ENUM as PG_ENUM
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from budapp.commons.constants import (
    CredentialTypeEnum,
    ModalityEnum,
    ModelProviderTypeEnum,
    ModelTemplateTypeEnum,
)
from budapp.commons.database import Base


class Model(Base):
    """Model for a AI model."""

    __tablename__ = "model"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=True)
    tags: Mapped[list[dict]] = mapped_column(JSONB, nullable=True)
    tasks: Mapped[list[dict]] = mapped_column(JSONB, nullable=True)
    author: Mapped[str] = mapped_column(String, nullable=True)
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
    created_by: Mapped[UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    endpoints: Mapped[list["Endpoint"]] = relationship(back_populates="model")
    # benchmarks: Mapped[list["Benchmark"]] = relationship(back_populates="model")

    created_user: Mapped["User"] = relationship(back_populates="created_models", foreign_keys=[created_by])

    paper_published: Mapped[List["PaperPublished"]] = relationship("PaperPublished", back_populates="model")
    model_licenses: Mapped["ModelLicenses"] = relationship("ModelLicenses", back_populates="model")

class PaperPublished(Base):
    """Model for Paper Published."""

    __tablename__ = "paper_published"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String, nullable=True)
    url: Mapped[str] = mapped_column(String, nullable=True)
    model_id: Mapped[UUID] = mapped_column(ForeignKey("model.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    model: Mapped["Model"] = relationship("Model", back_populates="paper_published")

class ModelLicenses(Base):
    """Model for a AI model licenses."""

    __tablename__ = "model_licenses"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=True)
    path: Mapped[str] = mapped_column(String, nullable=True)
    model_id: Mapped[UUID] = mapped_column(ForeignKey("model.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    model: Mapped["Model"] = relationship("Model", back_populates="model_licenses")

class Provider(Base):
    """Model for a AI model provider."""

    __tablename__ = "provider"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(
        Enum(
            CredentialTypeEnum,
            name="credential_type_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    description: Mapped[str] = mapped_column(String, nullable=True)
    icon: Mapped[Optional[str]] = mapped_column(String, nullable=False)


class CloudModel(Base):
    """Model for a AI cloud model."""

    __tablename__ = "cloud_model"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=True)
    tags: Mapped[list[dict]] = mapped_column(JSONB, nullable=True)
    tasks: Mapped[list[dict]] = mapped_column(JSONB, nullable=True)
    author: Mapped[str] = mapped_column(String, nullable=True)
    model_size: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    icon: Mapped[Optional[str]] = mapped_column(String, nullable=False)
    github_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    huggingface_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    website_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    modality: Mapped[str] = mapped_column(
        PG_ENUM(
            ModalityEnum,
            name="modality_enum",
            values_callable=lambda x: [e.value for e in x],
            create_type=False,
        ),
        nullable=False,
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    provider_type: Mapped[str] = mapped_column(
        PG_ENUM(
            ModelProviderTypeEnum,
            name="model_provider_type_enum",
            values_callable=lambda x: [e.value for e in x],
            create_type=False,
        ),
        nullable=False,
    )
    uri: Mapped[str] = mapped_column(String, nullable=False)
    provider_id: Mapped[UUID] = mapped_column(ForeignKey("provider.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelTemplate(Base):
    """Model template model"""

    __tablename__ = "model_template"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    icon: Mapped[str] = mapped_column(String, nullable=False)
    template_type: Mapped[str] = mapped_column(
        Enum(
            ModelTemplateTypeEnum,
            name="template_type_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        unique=True,
    )
    avg_sequence_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    avg_context_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    per_session_tokens_per_sec: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), nullable=True
    )
    ttft: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=True)
    e2e_latency: Mapped[list[int]] = mapped_column(ARRAY(Integer), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
