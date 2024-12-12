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
from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, ForeignKey, Integer, String, Uuid
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.dialects.postgresql import ENUM as PG_ENUM
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import false as sa_false

from budapp.commons.constants import (
    BaseModelRelationEnum,
    CredentialTypeEnum,
    ModalityEnum,
    ModelProviderTypeEnum,
    ModelSecurityScanStatusEnum,
    StatusEnum,
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
    icon: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    github_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    huggingface_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    bud_verified: Mapped[bool] = mapped_column(Boolean, default=False, server_default=sa_false())
    scan_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=True)
    eval_verified: Mapped[bool] = mapped_column(Boolean, default=False, server_default=sa_false())
    strengths: Mapped[list[str]] = mapped_column(PG_ARRAY(String), nullable=True)
    limitations: Mapped[list[str]] = mapped_column(PG_ARRAY(String), nullable=True)
    languages: Mapped[list[str]] = mapped_column(PG_ARRAY(String), nullable=True)
    use_cases: Mapped[list[str]] = mapped_column(PG_ARRAY(String), nullable=True)
    minimum_requirements: Mapped[dict] = mapped_column(JSONB, nullable=True)
    examples: Mapped[list[dict]] = mapped_column(JSONB, nullable=True)
    base_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    base_model_relation: Mapped[str] = mapped_column(
        Enum(
            BaseModelRelationEnum,
            name="base_model_relation_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=True,
    )
    model_type: Mapped[str] = mapped_column(String, nullable=True)
    family: Mapped[str] = mapped_column(String, nullable=True)
    num_layers: Mapped[int] = mapped_column(Integer, nullable=True)
    hidden_size: Mapped[int] = mapped_column(BigInteger, nullable=True)
    context_length: Mapped[int] = mapped_column(BigInteger, nullable=True)
    torch_dtype: Mapped[str] = mapped_column(String, nullable=True)
    architecture: Mapped[dict] = mapped_column(JSONB, nullable=True)
    website_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model_status: Mapped[str] = mapped_column(
        Enum(
            StatusEnum,
            name="status_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=StatusEnum.ACTIVE,
    )
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
    local_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    provider_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("provider.id"), nullable=True)
    created_by: Mapped[UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    endpoints: Mapped[list["Endpoint"]] = relationship(back_populates="model")
    # benchmarks: Mapped[list["Benchmark"]] = relationship(back_populates="model")
    created_user: Mapped["User"] = relationship(back_populates="created_models", foreign_keys=[created_by])
    paper_published: Mapped[List["PaperPublished"]] = relationship("PaperPublished", back_populates="model")
    model_licenses: Mapped["ModelLicenses"] = relationship("ModelLicenses", back_populates="model")
    provider: Mapped[Optional["Provider"]] = relationship("Provider", back_populates="models")
    model_security_scan_result: Mapped["ModelSecurityScanResult"] = relationship(
        "ModelSecurityScanResult", back_populates="model"
    )


class PaperPublished(Base):
    """Model for Paper Published."""

    __tablename__ = "paper_published"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String, nullable=True)
    authors: Mapped[list[str]] = mapped_column(PG_ARRAY(String), nullable=True)
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
    url: Mapped[str] = mapped_column(String, nullable=True)
    path: Mapped[str] = mapped_column(String, nullable=True)
    faqs: Mapped[list[dict]] = mapped_column(JSONB, nullable=True)
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

    models: Mapped[Optional[list["Model"]]] = relationship("Model", back_populates="provider")
    cloud_models: Mapped[list["CloudModel"]] = relationship("CloudModel", back_populates="provider")


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
    github_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    huggingface_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    website_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model_status: Mapped[str] = mapped_column(
        Enum(
            StatusEnum,
            name="status_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=StatusEnum.ACTIVE,
    )
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
    is_present_in_model: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    provider: Mapped[Optional["Provider"]] = relationship("Provider", back_populates="cloud_models")


class ModelSecurityScanResult(Base):
    """Model for a AI model security scan result."""

    __tablename__ = "model_security_scan_result"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    model_id: Mapped[UUID] = mapped_column(ForeignKey("model.id"), nullable=False)
    status: Mapped[str] = mapped_column(
        PG_ENUM(
            ModelSecurityScanStatusEnum,
            name="model_security_scan_status_enum",
            values_callable=lambda x: [e.value for e in x],
            create_type=False,
        ),
        nullable=False,
    )
    total_issues: Mapped[int] = mapped_column(Integer, nullable=False)
    total_scanned_files: Mapped[int] = mapped_column(Integer, nullable=False)
    total_skipped_files: Mapped[int] = mapped_column(Integer, nullable=False)
    scanned_files: Mapped[list[str]] = mapped_column(PG_ARRAY(String), nullable=False)
    low_severity_count: Mapped[int] = mapped_column(Integer, nullable=False)
    medium_severity_count: Mapped[int] = mapped_column(Integer, nullable=False)
    high_severity_count: Mapped[int] = mapped_column(Integer, nullable=False)
    critical_severity_count: Mapped[int] = mapped_column(Integer, nullable=False)
    model_issues: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    model: Mapped["Model"] = relationship("Model", back_populates="model_security_scan_result")
