from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from budmicroframe.shared.psql_service import CRUDMixin, PSQLBase, TimestampMixin
from sqlalchemy import Enum, ForeignKey, Integer, String, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..commons.constants import BenchmarkStatusEnum


if TYPE_CHECKING:
    from ..cluster_ops.models import Cluster
    from ..model_ops.models import Model
    from ..user_ops.models import User


class BenchmarkSchema(PSQLBase, TimestampMixin):
    """Benchmark model.

    model_id and cluster_id are kept nullable True, because
    we don't want model delete or cluster delete to have any
    effect on benchmark data.
    """

    __tablename__ = "benchmark"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    bud_cluster_benchmark_id: Mapped[UUID] = mapped_column(Uuid, nullable=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    model_id: Mapped[UUID] = mapped_column(ForeignKey("model.id"), nullable=True)
    cluster_id: Mapped[UUID] = mapped_column(ForeignKey("cluster.id"), nullable=True)
    nodes: Mapped[list[dict]] = mapped_column(JSONB, nullable=True)
    concurrency: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum(
            BenchmarkStatusEnum,
            name="benchmark_status_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    reason: Mapped[str] = mapped_column(String, nullable=True)

    model: Mapped["Model"] = relationship("Model", back_populates="benchmarks")
    cluster: Mapped["Cluster"] = relationship("Cluster", back_populates="benchmarks")
    user: Mapped["User"] = relationship("User", back_populates="benchmarks")


class BenchmarkCRUD(CRUDMixin[BenchmarkSchema, None, None]):
    __model__ = BenchmarkSchema

    def __init__(self):
        """Initialize benchmark crud methods."""
        super().__init__(model=self.__model__)