from typing import TYPE_CHECKING, Dict, List, Optional
from uuid import UUID, uuid4

from budmicroframe.shared.psql_service import CRUDMixin, PSQLBase, TimestampMixin
from sqlalchemy import Enum, ForeignKey, Integer, String, Uuid
from sqlalchemy import and_, asc, desc, distinct, func, or_, select, case, literal, cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..commons.constants import BenchmarkStatusEnum

from ..model_ops.models import Model
from ..cluster_ops.models import Cluster as ClusterModel


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
    tags: Mapped[list[dict]] = mapped_column(JSONB, nullable=True)
    description: Mapped[str] = mapped_column(String, nullable=True)
    eval_with: Mapped[str] = mapped_column(String, nullable=True)
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
    result: Mapped[dict] = mapped_column(JSONB, nullable=True)

    model: Mapped["Model"] = relationship("Model", back_populates="benchmarks")
    cluster: Mapped["Cluster"] = relationship("Cluster", back_populates="benchmarks")
    user: Mapped["User"] = relationship("User", back_populates="benchmarks")


class BenchmarkCRUD(CRUDMixin[BenchmarkSchema, None, None]):
    __model__ = BenchmarkSchema

    def __init__(self):
        """Initialize benchmark crud methods."""
        super().__init__(model=self.__model__)

    async def fetch_many_with_search(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Optional[Dict] = None,
        order_by: Optional[List] = None,
        search: bool = False,
    ):
        """Fetch many with search."""
        filters = filters or {}
        order_by = order_by or []

        translated_filters = filters.copy()
        translated_order_by = order_by.copy()
        for field in translated_filters:
            if field in ["model_name", "cluster_name"]:
                filters.pop(field)

        print(translated_filters)
        print(translated_order_by)


        await self.validate_fields(self.model, filters)

        # explicit conditions for order by model_name, cluster_name, modality
        explicit_conditions = []
        for field in translated_order_by:
            if field[0] == "model_name":
                sorting_stmt = await self.generate_sorting_stmt(
                    Model,
                    [
                        ("name", field[1]),
                    ],
                )
                explicit_conditions.extend(sorting_stmt)
                order_by.remove(field)
            elif field[0] == "cluster_name":
                sorting_stmt = await self.generate_sorting_stmt(
                    ClusterModel,
                    [
                        ("name", field[1]),
                    ],
                )
                explicit_conditions.extend(sorting_stmt)
                order_by.remove(field)


        if search:
            search_conditions = []
            for field, value in translated_filters.items():
                if field == "model_name":
                    search_conditions.extend(await self.generate_search_stmt(Model, {"name": value}))
                elif field == "cluster_name":
                    search_conditions.extend(await self.generate_search_stmt(ClusterModel, {"name": value}))
            search_conditions.extend(await self.generate_search_stmt(self.model, filters))

            stmt = (
                select(self.model)
                .join(Model)
                .join(ClusterModel)
                .filter(and_(*search_conditions))
            )
            count_stmt = (
                select(func.count())
                .select_from(self.model)
                .join(Model)
                .join(ClusterModel)
                .filter(and_(*search_conditions))
            )
        else:
            stmt = select(BenchmarkSchema).join(Model).join(ClusterModel)
            count_stmt = select(func.count()).select_from(self.model).join(Model).join(ClusterModel)
            for key, value in translated_filters.items():
                if key == "model_name":
                    stmt = stmt.filter(Model.name == value)
                    count_stmt = count_stmt.filter(Model.name == value)
                elif key == "cluster_name":
                    stmt = stmt.filter(ClusterModel.name == value)
                    count_stmt = count_stmt.filter(ClusterModel.name == value)
            for key, value in filters.items():
                stmt = stmt.filter(getattr(self.model, key) == value)
                count_stmt = count_stmt.filter(getattr(self.model, key) == value)

        # Calculate count before applying limit and offset
        count = self.execute_scalar(count_stmt)

        # Apply limit and offset
        stmt = stmt.limit(limit).offset(offset)

        # Apply sorting
        if translated_order_by:
            sort_conditions = await self.generate_sorting_stmt(self.model, order_by)
            # Extend sort conditions with explicit conditions
            sort_conditions.extend(explicit_conditions)
            stmt = stmt.order_by(*sort_conditions)

        print(stmt)

        result = self.scalars_all(stmt)

        return result, count
