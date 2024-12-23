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

"""The cluster ops package, containing essential business logic, services, and routing configurations for the cluster ops."""

import json
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String, Uuid
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship

from budapp.commons.constants import ClusterStatusEnum
from budapp.commons.database import Base


class Cluster(Base):
    """Cluster model."""

    __tablename__ = "cluster"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    ingress_url: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum(
            ClusterStatusEnum,
            name="cluster_status_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    icon: Mapped[str] = mapped_column(String, nullable=False)
    cpu_count: Mapped[int] = mapped_column(Integer, default=0)
    gpu_count: Mapped[int] = mapped_column(Integer, default=0)
    hpu_count: Mapped[int] = mapped_column(Integer, default=0)
    cpu_total_workers: Mapped[int] = mapped_column(Integer, default=0)
    cpu_available_workers: Mapped[int] = mapped_column(Integer, default=0)
    gpu_total_workers: Mapped[int] = mapped_column(Integer, default=0)
    gpu_available_workers: Mapped[int] = mapped_column(Integer, default=0)
    hpu_total_workers: Mapped[int] = mapped_column(Integer, default=0)
    hpu_available_workers: Mapped[int] = mapped_column(Integer, default=0)
    reason: Mapped[str] = mapped_column(String, nullable=True)
    created_by: Mapped[UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    cluster_id: Mapped[UUID] = mapped_column(Uuid, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status_sync_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    endpoints: Mapped[list["Endpoint"]] = relationship(
        "Endpoint",
        back_populates="cluster",
    )
    # benchmarks: Mapped[list["Benchmark"]] = relationship(
    #     "Benchmark",
    #     back_populates="cluster",
    # )
    created_user: Mapped["User"] = relationship(back_populates="created_clusters", foreign_keys=[created_by])

    @hybrid_property
    def kubernetes_info_dict(self):
        if not self.kubernetes_metadata:
            return {}
        return json.loads(self.kubernetes_metadata)
