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

"""The endpoint ops package, containing essential business logic, services, and routing configurations for the endpoint ops."""

import json
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String, Uuid
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship

from budapp.cluster_ops.models import Cluster
from budapp.commons.constants import EndpointStatusEnum
from budapp.commons.database import Base
from budapp.model_ops.models import Model
from budapp.project_ops.models import Project


class Endpoint(Base):
    """Endpoint model."""

    __tablename__ = "endpoint"
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    project_id: Mapped[UUID] = mapped_column(ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    model_id: Mapped[UUID] = mapped_column(ForeignKey("model.id", ondelete="CASCADE"), nullable=False)
    cache_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_config: Mapped[str] = mapped_column(String, nullable=True)
    cluster_id: Mapped[UUID] = mapped_column(ForeignKey("cluster.id", ondelete="CASCADE"), nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    namespace: Mapped[str] = mapped_column(String, nullable=False)
    replicas: Mapped[int] = mapped_column(Integer, nullable=False)
    created_by: Mapped[UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    modified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status: Mapped[str] = mapped_column(
        Enum(
            EndpointStatusEnum,
            name="endpoint_status_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )
    status_sync_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    model: Mapped[Model] = relationship("Model", back_populates="endpoints", foreign_keys=[model_id])
    # worker: Mapped[Worker] = relationship(
    #     "Worker",
    #     back_populates="endpoint",
    #     cascade="all, delete",
    #     passive_deletes=True,
    # )
    project: Mapped[Project] = relationship("Project", back_populates="endpoints", foreign_keys=[project_id])

    cluster: Mapped[Cluster] = relationship("Cluster", back_populates="endpoints", foreign_keys=[cluster_id])
    created_user: Mapped["User"] = relationship(back_populates="created_endpoints", foreign_keys=[created_by])

    @hybrid_property
    def cache_config_dict(self):
        if not self.cache_config:
            return {}
        return json.loads(self.cache_config)

    def to_dict(self):
        return {
            "id": str(self.id),
            "is_active": self.is_active,
            "project_id": str(self.project_id),
            "model_id": str(self.model_id),
            "cache_enabled": self.cache_enabled,
            "cache_config": self.cache_config_dict,
            "cluster_id": str(self.cluster_id),
            "url": self.url,
            "namespace": self.namespace,
            "replicas": self.replicas,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            id=data.get("id"),
            is_active=data.get("is_active"),
            project_id=UUID(data.get("project_id")),
            model_id=UUID(data.get("model_id")),
            cache_enabled=data.get("cache_enabled"),
            cache_config=json.dumps(data.get("cache_config")),
            cluster_id=UUID(data.get("cluster_id")),
            url=data.get("url"),
            namespace=data.get("namespace"),
            replicas=data.get("replicas"),
            created_at=datetime.fromisoformat(data.get("created_at")),
            modified_at=datetime.fromisoformat(data.get("modified_at")),
        )
