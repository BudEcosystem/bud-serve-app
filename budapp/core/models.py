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

"""The core package, containing essential business logic, services, and routing configurations for the microservices."""

from uuid import UUID, uuid4
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Uuid, ARRAY, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from budapp.commons.constants import WorkflowStatusEnum, ModelTemplateTypeEnum
from budapp.commons.database import Base


class Icon(Base):
    """Icon model."""

    __tablename__ = "icon"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, index=True, nullable=False)
    file_path: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    category: Mapped[str] = mapped_column(String, index=True, nullable=False)


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