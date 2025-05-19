# budapp/eval_ops/models.py

from enum import Enum as PyEnum
from uuid import uuid4

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ENUM as PG_ENUM
from sqlalchemy.dialects.postgresql import JSONB, NUMERIC
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from budapp.commons.database import Base, TimestampMixin


# ------------------------ Enums ------------------------

class EvaluationStatusEnum(PyEnum):
    ACTIVE = "active"
    DELETED = "deleted"


class EvaluationRunStatusEnum(PyEnum):
    DELETED     = "deleted"
    RUNNING     = "running"
    COMPLETED   = "completed"
    FAILED      = "failed"
    PENDING     = "pending"
    CANCELLED   = "cancelled"
    SKIPPED     = "skipped"
    SUMMARIZING = "summarizing"


# ------------------------ Core Tables ------------------------

class Evaluation(Base, TimestampMixin):
    __tablename__ = "evaluation"

    id:           Mapped[uuid4] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name:         Mapped[str]   = mapped_column(String, nullable=False)
    description:  Mapped[str]   = mapped_column(String, nullable=True)
    status:       Mapped[str]   = mapped_column(
        PG_ENUM(*[e.value for e in EvaluationStatusEnum], name="evaluation_status_enum"),
        nullable=False,
    )
    created_by:   Mapped[uuid4] = mapped_column(ForeignKey("user.id"), nullable=False)
    project_id:   Mapped[uuid4] = mapped_column(ForeignKey("project.id"), nullable=False)

    runs = relationship("Run", back_populates="evaluation", cascade="all, delete-orphan")
    run_dataset_configs = relationship(
        "EvaluationRunDatasetConfig",
        back_populates="evaluation",
        cascade="all, delete-orphan"
    )


class Run(Base, TimestampMixin):
    __tablename__ = "evaluation_runs"

    id:            Mapped[uuid4] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    evaluation_id: Mapped[uuid4] = mapped_column(ForeignKey("evaluation.id"), nullable=False)
    name:          Mapped[str]   = mapped_column(String, nullable=False)
    description:   Mapped[str]   = mapped_column(String, nullable=True)
    status:        Mapped[str]   = mapped_column(
        PG_ENUM(*[e.value for e in EvaluationRunStatusEnum], name="evaluation_run_status_enum"),
        nullable=False,
    )

    evaluation = relationship("Evaluation", back_populates="runs")
    dataset_configs = relationship(
        "EvaluationRunDatasetConfig",
        back_populates="run",
        cascade="all, delete-orphan"
    )
    metrics     = relationship("EvalMetric",     back_populates="run",      cascade="all, delete-orphan")
    raw_results = relationship("EvalRawResult",  back_populates="run",      cascade="all, delete-orphan")


class EvaluationRunDatasetConfig(Base, TimestampMixin):
    __tablename__ = "evaluation_run_dataset_config"

    id:            Mapped[uuid4] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id:        Mapped[uuid4] = mapped_column(ForeignKey("evaluation_runs.id"), nullable=False)
    evaluation_id: Mapped[uuid4] = mapped_column(ForeignKey("evaluation.id"),      nullable=False)

    run        = relationship("Run",        back_populates="dataset_configs")
    evaluation = relationship("Evaluation", back_populates="run_dataset_configs")


# ------------------------ Lookup Tables ------------------------

class Model(Base, TimestampMixin):
    __tablename__ = "eval_models"

    id:          Mapped[uuid4] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name:        Mapped[str]   = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str]   = mapped_column(String, nullable=True)

    metrics     = relationship("EvalMetric",    back_populates="model",      cascade="all, delete-orphan")
    raw_results = relationship("EvalRawResult", back_populates="model",      cascade="all, delete-orphan")


class Trait(Base, TimestampMixin):
    __tablename__ = "eval_traits"

    id:          Mapped[uuid4] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name:        Mapped[str]   = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str]   = mapped_column(String, nullable=True)

    datasets = relationship("Dataset", back_populates="trait", cascade="all, delete-orphan")


class Dataset(Base, TimestampMixin):
    __tablename__ = "eval_datasets"
    __table_args__ = (UniqueConstraint('trait_id', 'name', name='uq_dataset_trait_name'),)

    id:          Mapped[uuid4] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    trait_id:    Mapped[uuid4] = mapped_column(ForeignKey("eval_traits.id"),  nullable=False)
    name:        Mapped[str]   = mapped_column(String, nullable=False)
    description: Mapped[str]   = mapped_column(String, nullable=True)

    trait    = relationship("Trait",           back_populates="datasets")
    versions = relationship("DatasetVersion",  back_populates="dataset",  cascade="all, delete-orphan")


class DatasetVersion(Base, TimestampMixin):
    __tablename__ = "eval_dataset_versions"
    __table_args__ = (UniqueConstraint('dataset_id', 'version', name='uq_datasetversion_dataset_version'),)

    id:             Mapped[uuid4] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    dataset_id:     Mapped[uuid4] = mapped_column(ForeignKey("eval_datasets.id"),          nullable=False)
    version:        Mapped[str]   = mapped_column(String, nullable=False)
    meta:           Mapped[dict]  = mapped_column(JSONB, nullable=True)

    dataset      = relationship("Dataset",        back_populates="versions")
    metrics      = relationship("EvalMetric",     back_populates="dataset_version", cascade="all, delete-orphan")
    raw_results  = relationship("EvalRawResult",  back_populates="dataset_version", cascade="all, delete-orphan")


# ------------------------ Evaluation Results ------------------------

class EvalMetric(Base):
    __tablename__ = "eval_metrics"
    __table_args__ = (
        UniqueConstraint(
            'run_id', 'dataset_version_id', 'model_id', 'metric_name', 'mode',
            name='uq_evalmetrics_combination'
        ),
    )

    id:                 Mapped[uuid4] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id:             Mapped[uuid4] = mapped_column(ForeignKey("evaluation_runs.id"),         nullable=False)
    dataset_version_id: Mapped[uuid4] = mapped_column(ForeignKey("eval_dataset_versions.id"),  nullable=False)
    model_id:           Mapped[uuid4] = mapped_column(ForeignKey("eval_models.id"),            nullable=False)
    metric_name:        Mapped[str]   = mapped_column(String, nullable=False)
    mode:               Mapped[str]   = mapped_column(String, nullable=False)
    metric_value:       Mapped[float] = mapped_column(NUMERIC(6, 2), nullable=False)

    run             = relationship("Run",            back_populates="metrics")
    dataset_version = relationship("DatasetVersion", back_populates="metrics")
    model           = relationship("Model",          back_populates="metrics")


class EvalRawResult(Base, TimestampMixin):
    __tablename__ = "eval_raw_results"
    __table_args__ = (
        UniqueConstraint('run_id', 'dataset_version_id', 'model_id', name='uq_evalrawresults_run_dv_model'),
    )

    id:                  Mapped[uuid4] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id:              Mapped[uuid4] = mapped_column(ForeignKey("evaluation_runs.id"),         nullable=False)
    dataset_version_id:  Mapped[uuid4] = mapped_column(ForeignKey("eval_dataset_versions.id"),  nullable=False)
    model_id:            Mapped[uuid4] = mapped_column(ForeignKey("eval_models.id"),            nullable=False)
    preview_results:     Mapped[dict]  = mapped_column(JSONB, nullable=False)
    full_results_uri:    Mapped[str]   = mapped_column(Text, nullable=True)

    run             = relationship("Run",            back_populates="raw_results")
    dataset_version = relationship("DatasetVersion", back_populates="raw_results")
    model           = relationship("Model",          back_populates="raw_results")