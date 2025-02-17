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

"""The models package, containing the database models for the user ops."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Enum, Integer, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from budapp.cluster_ops.models import Cluster
from budapp.commons.constants import UserRoleEnum, UserStatusEnum
from budapp.commons.database import Base, TimestampMixin
from budapp.endpoint_ops.models import Endpoint
from budapp.model_ops.models import Model
from budapp.project_ops.models import Project, project_user_association


class User(Base, TimestampMixin):
    """User model."""

    __tablename__ = "user"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    auth_id: Mapped[UUID] = mapped_column(Uuid, unique=True, default=uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    role: Mapped[str] = mapped_column(
        Enum(
            UserRoleEnum,
            name="user_role_enum",
            values_callable=lambda x: [e.value for e in x],
        )
    )
    status: Mapped[str] = mapped_column(
        Enum(
            UserStatusEnum,
            name="user_status_enum",
            values_callable=lambda x: [e.value for e in x],
        ),
        default=UserStatusEnum.INVITED.value,
    )
    password: Mapped[str] = mapped_column(String, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    is_reset_password: Mapped[bool] = mapped_column(Boolean, default=True)
    color: Mapped[str] = mapped_column(String, nullable=False)
    first_login: Mapped[bool] = mapped_column(Boolean, default=True)
    is_subscriber: Mapped[bool] = mapped_column(Boolean, default=False)
    reset_password_attempt: Mapped[int] = mapped_column(Integer, default=0)

    permission: Mapped["Permission"] = relationship(back_populates="user")  # one-to-one
    created_models: Mapped[list[Model]] = relationship(back_populates="created_user")

    # TODO: uncomment when implement individual fields
    # benchmarks: Mapped[list["Benchmark"]] = relationship(back_populates="user")
    # benchmark_results: Mapped[list["BenchmarkResult"]] = relationship(back_populates="user")
    projects: Mapped[list[Project]] = relationship(secondary=project_user_association, back_populates="users")
    # project_permissions: Mapped[list[ProjectPermission]] = relationship(
    #     back_populates="user"
    # )
    created_projects: Mapped[list[Project]] = relationship(back_populates="created_user")
    created_clusters: Mapped[list[Cluster]] = relationship(back_populates="created_user")
    created_endpoints: Mapped[list[Endpoint]] = relationship(back_populates="created_user")
