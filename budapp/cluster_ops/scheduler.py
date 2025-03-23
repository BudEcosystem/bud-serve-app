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

"""The model ops scheduler. Contains business logic for model ops."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import List, Optional
from uuid import UUID

import aiohttp
from budmicroframe.shared.dapr_service import DaprService
from fastapi import status
from sqlalchemy.orm import Session

from ..commons import logging
from ..commons.config import app_settings
from ..commons.constants import (
    BUD_INTERNAL_WORKFLOW,
    ModelProviderTypeEnum,
    ModelStatusEnum,
)
from ..commons.database import engine
from ..commons.exceptions import ClientException
from ..commons.schemas import BudNotificationMetadata
from ..model_ops.crud import ModelDataManager
from ..model_ops.models import Model
from .crud import ModelClusterRecommendedDataManager
from .models import ModelClusterRecommended
from .schemas import BudSimulatorRequest


logger = logging.get_logger(__name__)


RECOMMENDED_CLUSTER_SCHEDULER_INTERVAL_HOURS = 24


class RecommendedClusterScheduler:
    """The recommended cluster scheduler. Contains business logic for recommended cluster per model."""

    @staticmethod
    def get_models(model_id: Optional[UUID] = None) -> List[Model]:
        """Get the models."""
        with Session(engine) as session:
            if model_id:
                db_model = asyncio.run(
                    ModelDataManager(session).retrieve_by_fields(
                        {"id": model_id, "status": ModelStatusEnum.ACTIVE}, missing_ok=True
                    )
                )
                if db_model:
                    return [db_model]
                else:
                    logger.error("Model with id %s not found in database", model_id)
                    return []
            else:
                db_model_cluster_recommended = asyncio.run(
                    ModelClusterRecommendedDataManager(session).get_all_by_fields(ModelClusterRecommended, {})
                )
                if db_model_cluster_recommended:
                    current_time = datetime.now(UTC)
                    older_than = current_time - timedelta(hours=RECOMMENDED_CLUSTER_SCHEDULER_INTERVAL_HOURS)
                    logger.debug("Getting model cluster recommendation older than %s", older_than)
                    db_model_cluster_recommended = asyncio.run(
                        ModelClusterRecommendedDataManager(session).get_stale_model_recommendation(older_than)
                    )
                    if db_model_cluster_recommended:
                        return [db_model_cluster_recommended.model_id]
                    else:
                        logger.error("No stale model cluster recommendation found in db")
                        return []
                else:
                    db_models = asyncio.run(
                        ModelDataManager(session).get_all_by_fields(Model, {"status": ModelStatusEnum.ACTIVE})
                    )
                    if db_models:
                        return [db_models[0]]
                    else:
                        logger.error("No active models found in database")
                        return []

    def execute_cluster_recommendation(self, model_id: Optional[UUID] = None) -> None:
        """Execute the cluster recommendation."""
        db_models = self.get_models(model_id)
        logger.debug("Found %s models to execute cluster recommendation", len(db_models))

        if not db_models:
            logger.debug("All models are up to date with recommended clusters")
            return

        for db_model in db_models:
            logger.debug("Executing cluster recommendation for model %s", db_model.id)

            # Create Bud Simulator Request for cloud/local models
            workflow_id = str(uuid.uuid4())
            bud_simulator_request = self._build_bud_simulator_request(db_model)
            bud_simulator_request.notification_metadata = BudNotificationMetadata(
                workflow_id=workflow_id,
                name=BUD_INTERNAL_WORKFLOW,
                subscriber_ids=str(db_model.created_by),
            )

            try:
                response = asyncio.run(self._perform_bud_simulator_request(bud_simulator_request))
            except ClientException:
                logger.error("Failed to initiate bud simulator workflow %s", workflow_id)
                continue

            if isinstance(response, dict) and "workflow_id" in response:
                logger.debug("Successfully initiated bud simulator workflow %s", workflow_id)

                # Save workflow details in state store
                recommended_cluster_scheduler_state = {
                    str(workflow_id): {
                        "model_id": str(db_model.id),
                    }
                }
                state_store_key = "recommended_cluster_scheduler_state"

                try:
                    dapr_service = DaprService()
                    dapr_service.save_to_statestore(
                        store_name=app_settings.statestore_name,
                        key=state_store_key,
                        value=recommended_cluster_scheduler_state,
                    )
                except Exception as e:
                    logger.exception("Failed to save state store %s", e)
                    continue

                logger.debug("data pushed to dapr state store %s", recommended_cluster_scheduler_state)
            else:
                logger.error("Failed to initiate bud simulator workflow")

    @staticmethod
    async def _perform_bud_simulator_request(bud_simulator_request: BudSimulatorRequest) -> None:
        """Perform the bud simulator request."""
        bud_simulator_endpoint = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_model_app_id}/method/simulator/run"
        )

        payload = bud_simulator_request.model_dump()

        logger.debug(f"Performing bud simulator request {payload}")
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(bud_simulator_endpoint, json=payload) as response
            ):
                response_data = await response.json()
                if response.status != 200:
                    logger.error(f"Failed to fetch bud simulator response: {response.status} {response_data}")
                    raise ClientException(
                        "Failed to fetch bud simulator response", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )

                logger.debug(f"Successfully fetched bud simulator response{response_data}")
                return response_data
        except Exception as e:
            logger.exception(f"Failed to send bud simulator request: {e}")
            raise ClientException(
                "Failed to send bud simulator request", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from e

    @staticmethod
    def _build_bud_simulator_request(db_model: Model) -> BudSimulatorRequest:
        """Build the bud simulator request.

        Args:
            db_model (Model): The model to build the bud simulator request for.

        Returns:
            BudSimulatorRequest: The bud simulator request.
        """
        if db_model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
            bud_simulator_request = BudSimulatorRequest(
                pretrained_model_uri=db_model.uri,
                input_tokens=50,  # context length
                output_tokens=100,  # sequence length
                concurrency=10,  # concurrent requests
                target_throughput_per_user=0,  # minimum ttft
                target_ttft=0,  # maximum per session tokens/second
                target_e2e_latency=0,  # min e2e latency
                notification_metadata=None,
                source_topic=app_settings.source_topic,
                is_proprietary_model=True,
            )
        else:
            bud_simulator_request = BudSimulatorRequest(
                pretrained_model_uri=db_model.local_path,
                input_tokens=50,
                output_tokens=100,
                concurrency=10,
                target_throughput_per_user=25,
                target_ttft=300,
                target_e2e_latency=10,
                notification_metadata=None,
                source_topic=app_settings.source_topic,
                is_proprietary_model=False,
            )

        return bud_simulator_request


if __name__ == "__main__":
    recommended_cluster_scheduler = RecommendedClusterScheduler()
    recommended_cluster_scheduler.execute_cluster_recommendation()

    # python -m budapp.cluster_ops.scheduler
