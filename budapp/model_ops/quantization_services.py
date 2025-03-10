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

"""The model ops services. Contains business logic for quantiation in model ops."""

from typing import Any, Dict, List, Tuple
from uuid import UUID

import aiohttp

from budapp.commons import logging
from budapp.commons.config import app_settings
from budapp.commons.constants import (
    APP_ICONS,
    BUD_INTERNAL_WORKFLOW,
    ModelProviderTypeEnum,
    ModelStatusEnum,
    WorkflowTypeEnum,
)
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.workflow_ops.crud import WorkflowDataManager, WorkflowStepDataManager
from budapp.workflow_ops.models import Workflow as WorkflowModel
from budapp.workflow_ops.models import WorkflowStep as WorkflowStepModel
from budapp.workflow_ops.schemas import WorkflowUtilCreate
from budapp.workflow_ops.services import WorkflowService

from .crud import ModelDataManager, ProviderDataManager, QuantizationMethodDataManager
from .models import Model
from .models import Provider as ProviderModel, QuantizationMethod
from .schemas import QuantizeConfig, QuantizeModelWorkflowRequest, QuantizeModelWorkflowStepData


logger = logging.get_logger(__name__)

class QuantizationService(SessionMixin):
    """Quantization service."""

    async def get_quantization_methods(self, offset: int, limit: int, filters: Dict[str, Any] = {}, order_by: List[Tuple[str, str]] = [], search: bool = False) -> Tuple[List[QuantizationMethod], int]:
        """Get all quantization methods."""
        return await QuantizationMethodDataManager(self.session).get_all_quantization_methods(offset, limit, filters, order_by, search)

    async def quantize_model_workflow(self, current_user_id: UUID, request: QuantizeModelWorkflowRequest) -> None:
        """Quantize a model."""
        
        step_number = request.step_number
        workflow_id = request.workflow_id
        workflow_total_steps = request.workflow_total_steps
        trigger_workflow = request.trigger_workflow
        model_id = request.model_id
        quantized_model_name = request.quantized_model_name
        target_type = request.target_type
        target_device = request.target_device
        method = request.method
        weight_config = request.weight_config
        activation_config = request.activation_config
        current_step_number = step_number

        # Retrieve or create workflow
        workflow_create = WorkflowUtilCreate(
            workflow_type=WorkflowTypeEnum.LOCAL_MODEL_QUANTIZATION,
            title="Quantize Model",
            total_steps=workflow_total_steps,
            icon=APP_ICONS["general"]["model_mono"],
            tag="Model",
        )
        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id, workflow_create, current_user_id
        )


        #validate base model id
        if model_id:
            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model, {"id": model_id, "status": ModelStatusEnum.ACTIVE}
            )
            if db_model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
                raise ClientException("Quantization is only supported for local models")

            # Update icon on workflow
            if db_model.provider_type == ModelProviderTypeEnum.HUGGING_FACE:
                db_provider = await ProviderDataManager(self.session).retrieve_by_fields(
                    ProviderModel, {"id": db_model.provider_id}
                )
                model_icon = db_provider.icon
            else:
                model_icon = db_model.icon

            # Update title, icon on workflow
            db_workflow = await WorkflowDataManager(self.session).update_by_fields(
                db_workflow,
                {"title": db_model.name, "icon": model_icon},
            )

        # Prepare workflow step data
        workflow_step_data = QuantizeModelWorkflowStepData(
            model_id=model_id,
            quantized_model_name=quantized_model_name,
            target_type=target_type,
            target_device=target_device,
            method=method,
            weight_config=weight_config,
            activation_config=activation_config,
        ).model_dump(exclude_none=True, exclude_unset=True, mode="json")

        # Get workflow steps
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": db_workflow.id}
        )

        # For avoiding another db call for record retrieval, storing db object while iterating over db_workflow_steps
        db_current_workflow_step = None

        if db_workflow_steps:
            for db_step in db_workflow_steps:
                # Get current workflow step
                if db_step.step_number == current_step_number:
                    db_current_workflow_step = db_step

        if db_current_workflow_step:
            logger.info(f"Workflow {db_workflow.id} step {current_step_number} already exists")

            # Update workflow step data in db
            db_workflow_step = await WorkflowStepDataManager(self.session).update_by_fields(
                db_current_workflow_step,
                {"data": workflow_step_data},
            )
            logger.info(f"Workflow {db_workflow.id} step {current_step_number} updated")
        else:
            logger.info(f"Creating workflow step {current_step_number} for workflow {db_workflow.id}")

            # Insert step details in db
            db_workflow_step = await WorkflowStepDataManager(self.session).insert_one(
                WorkflowStepModel(
                    workflow_id=db_workflow.id,
                    step_number=current_step_number,
                    data=workflow_step_data,
                )
            )

        # Update workflow current step as the highest step_number
        db_max_workflow_step_number = max(step.step_number for step in db_workflow_steps) if db_workflow_steps else 0
        workflow_current_step = max(current_step_number, db_max_workflow_step_number)
        logger.info(f"The current step of workflow {db_workflow.id} is {workflow_current_step}")

        # This will ensure workflow step number is updated to the latest step number
        db_workflow = await WorkflowDataManager(self.session).update_by_fields(
            db_workflow,
            {"current_step": workflow_current_step},
        )
        #TODO: add validation for cluster_id
        
        # Perform simulation
        if method is not None and weight_config is not None and activation_config is not None:
            # Perform weight quantization
            pass

        # Trigger workflow
        if trigger_workflow:
            # query workflow steps again to get latest data
            db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
                {"workflow_id": db_workflow.id}
            )

            # Define the keys required for model extraction
            keys_of_interest = [
                "model_id",
                "quantized_model_name",
                "method",
                "weight_config",
                "activation_config",
                "cluster_id",
                "simulator_id",
            ]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            # Check if all required keys are present
            required_keys = ["model_id", "quantized_model_name", "method", "weight_config", "activation_config", "cluster_id", "simulator_id"]
            missing_keys = [key for key in required_keys if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data for add worker to deployment: {', '.join(missing_keys)}")

            # Get base model
            required_data["base_model_uri"] = db_model.local_path

            try:
                # Perform model quantization
                await self._perform_model_quantization(
                    current_step_number, required_data, db_workflow, current_user_id
                )
            except ClientException as e:
                raise e

        return db_workflow

    async def _perform_model_quantization(
        self, current_step_number: int, data: Dict, db_workflow: WorkflowModel, current_user_id: UUID
    ) -> None:
        """Perform weight quantization."""
        # Create request payload
        payload = {
            "quantization_config": {
                "weight": data["weight_config"],
                "activation": data["activation_config"],
            },
            "quantization_name": data["model_name"],
            "model": data["base_model_uri"],
            "simulator_id": data["simulator_id"],
            "cluster_id": str(data["cluster_id"]),
            "notification_metadata": {
                "name": BUD_INTERNAL_WORKFLOW,
                "subscriber_ids": str(current_user_id),
                "workflow_id": str(db_workflow.id),
            },
            "source_topic": f"{app_settings.source_topic}",
        }

        quantize_endpoint = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_cluster_app_id}/method/deployment/deploy-quantization"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(quantize_endpoint, json=payload) as response:
                    response_data = await response.json()
                    if response.status >= 400 or response_data.get("object") == "error":
                        raise ClientException("Unable to perform model quantization")

                    return response_data
        except ClientException as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to perform model quantization request: {e}")
            raise ClientException("Unable to perform model quantization") from e

