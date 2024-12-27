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

"""The crud package, containing essential business logic, services, and routing configurations for the metric ops."""

from typing import Any, Dict, List, Tuple
from uuid import UUID

from sqlalchemy import and_, func, or_, select, distinct

from budapp.cluster_ops.models import Cluster as ClusterModel
from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils
from budapp.model_ops.models import Model as Model
from budapp.commons.constants import (
    EndpointStatusEnum,
    ModelStatusEnum,
    ClusterStatusEnum,
    ModelProviderTypeEnum,
)
from ..endpoint_ops.models import Endpoint as EndpointModel
from ..project_ops.models import project_user_association

logger = logging.get_logger(__name__)


class MetricDataManager(DataManagerUtils):
    """Data manager for the Metric model."""

    async def get_dashboard_stats(self, user_id: UUID) -> Tuple[Dict[str, int]]:
        db_total_model_count = await self.get_count_by_fields(Model, fields={"status": ModelStatusEnum.ACTIVE})
        db_cloud_model_count = await self.get_count_by_fields(
            Model, fields={"status": ModelStatusEnum.ACTIVE, "provider_type": ModelProviderTypeEnum.CLOUD_MODEL}
        )
        db_local_model_count = await self.get_count_by_fields(
            Model,
            fields={"status": ModelStatusEnum.ACTIVE},
            exclude_fields={"provider_type": ModelProviderTypeEnum.CLOUD_MODEL},
        )
        db_total_endpoint_count = await self.get_count_by_fields(
            EndpointModel, fields={}, exclude_fields={"status": EndpointStatusEnum.DELETED}
        )
        db_running_endpoint_count = await self.get_count_by_fields(
            EndpointModel, fields={"status": EndpointStatusEnum.RUNNING}
        )

        db_total_clusters = await self.get_count_by_fields(
            ClusterModel, fields={}, exclude_fields={"status": ClusterStatusEnum.DELETED}
        )

        db_inactive_clusters = await self.get_count_by_fields(
            ClusterModel, fields={"status": ClusterStatusEnum.NOT_AVAILABLE}
        )

        projects_count_stmt = select(func.count(distinct(project_user_association.c.project_id))).where(
            project_user_association.c.user_id == user_id
        )
        db_total_projects = self.scalar_one_or_none(projects_count_stmt) or 0

        # Query to count unique users in the projects the user is part of
        users_count_stmt = select(func.count(distinct(project_user_association.c.user_id))).where(
            project_user_association.c.project_id.in_(
                select(project_user_association.c.project_id).where(project_user_association.c.user_id == user_id)
            )
        )
        db_total_project_users = self.scalar_one_or_none(users_count_stmt) or 0

        dashboard_stats = {
            "total_model_count": db_total_model_count,
            "cloud_model_count": db_cloud_model_count,
            "local_model_count": db_local_model_count,
            "total_projects": db_total_projects,
            "total_project_users": db_total_project_users,
            "total_endpoints_count": db_total_endpoint_count,
            "running_endpoints_count": db_running_endpoint_count,
            "total_clusters": db_total_clusters,
            "inactive_clusters": db_inactive_clusters,
        }

        return dashboard_stats
