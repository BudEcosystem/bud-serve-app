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

"""The eval ops package, containing essential business logic, services, and routing configurations for the eval ops."""

from enum import Enum as PyEnum
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, String, Uuid
from sqlalchemy.dialects.postgresql import ENUM as PG_ENUM
from sqlalchemy.orm import Mapped, mapped_column, relationship

from budapp.commons.database import Base, TimestampMixin
from budapp.user_ops.models import User  # Import the User model


class EvaluationStatusEnum(PyEnum):
    """Enumeration of evaluation statuses."""

    ACTIVE = "active"
    DELETED = "deleted"

class EvaluationRunStatusEnum(PyEnum):
    """Enumeration of evaluation run statuses."""

    DELETED = "deleted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    SUMMARIZING = "summarizing"


class Evaluation(Base, TimestampMixin):
    """Model for a evaluation."""

    __tablename__ = "evaluation"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(
        PG_ENUM(
            *[e.value for e in EvaluationStatusEnum],
            name="evaluation_status_enum",
        ),
    )
    created_by: Mapped[UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("project.id"), nullable=False)

    # created_user: Mapped[User] = relationship(back_populates="created_evaluations", foreign_keys=[created_by])

    runs = relationship("Run", back_populates="evaluation", cascade="all, delete-orphan")
    run_dataset_configs = relationship("EvaluationRunDatasetConfig", back_populates="evaluation", cascade="all, delete-orphan")


class Run(Base, TimestampMixin):
    """Model for a run."""

    __tablename__ = "evaluation_runs"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    evaluation_id: Mapped[UUID] = mapped_column(ForeignKey("evaluation.id"), nullable=False)
    evaluation: Mapped["Evaluation"] = relationship(back_populates="runs")

    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(
        PG_ENUM(
            *[e.value for e in EvaluationRunStatusEnum],
            name="evaluation_run_status_enum",
        ),
    )

    dataset_configs = relationship("EvaluationRunDatasetConfig", back_populates="run", cascade="all, delete-orphan")


class EvaluationRunDatasetConfig(Base, TimestampMixin):
    """Model for a evaluation run dataset config."""

    __tablename__ = "evaluation_run_dataset_config"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("evaluation_runs.id"), nullable=False)
    run: Mapped["Run"] = relationship(back_populates="dataset_configs")

    evaluation_id: Mapped[UUID] = mapped_column(ForeignKey("evaluation.id"), nullable=False)
    evaluation: Mapped["Evaluation"] = relationship(back_populates="run_dataset_configs")






