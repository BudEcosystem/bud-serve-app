import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from uuid import UUID

import aiohttp
from fastapi import status

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin

from ..cluster_ops.services import ClusterService
from ..commons.config import app_settings
from ..commons.constants import (
    APP_ICONS,
    BUD_INTERNAL_WORKFLOW,
    BudServeWorkflowStepEventName,
    NotificationTypeEnum,
    WorkflowStatusEnum,
    WorkflowTypeEnum,
    ModelStatusEnum,
    ClusterStatusEnum,
)
from ..commons.exceptions import ClientException, RedisException
from ..core.schemas import NotificationPayload, NotificationResult
from ..model_ops.services import ModelService, ModelServiceUtil
from ..shared.notification_service import BudNotifyService, NotificationBuilder
from ..workflow_ops.crud import WorkflowDataManager, WorkflowStepDataManager
from ..workflow_ops.models import Workflow as WorkflowModel
from ..workflow_ops.models import WorkflowStep as WorkflowStepModel
from ..workflow_ops.schemas import WorkflowUtilCreate
from ..workflow_ops.services import WorkflowService, WorkflowStepService
from .models import BenchmarkCRUD
from .schemas import RunBenchmarkWorkflowRequest, RunBenchmarkWorkflowStepData

from ..model_ops.crud import ModelDataManager
from ..model_ops.models import Model


logger = logging.get_logger(__name__)

class BenchmarkService(SessionMixin):
    """Benchmark service."""

    async def run_benchmark_workflow(self, current_user_id: UUID, request: RunBenchmarkWorkflowRequest):
        # Get request data
        step_number = request.step_number
        workflow_id = request.workflow_id
        workflow_total_steps = request.workflow_total_steps
        trigger_workflow = request.trigger_workflow

        # resources
        model_id = request.model_id
        cluster_id = request.cluster_id

        current_step_number = step_number

        # Retrieve or create workflow
        workflow_create = WorkflowUtilCreate(
            workflow_type=WorkflowTypeEnum.ADD_WORKER_TO_ENDPOINT,
            title="Add Worker to Deployment",
            total_steps=workflow_total_steps,
            icon=APP_ICONS["general"]["deployment_mono"],
            tag="Deployment",
        )
        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id, workflow_create, current_user_id
        )

        # Validate resources
        if model_id:
            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model, fields={"id": model_id}, exclude_fields={"status": ModelStatusEnum.DELETED}
            )
            if not db_model:
                raise ClientException("Model does not exist")

        if cluster_id:
            db_cluster = await ClusterService(self.session).retrieve_by_fields(
                db_cluster, fields={"id": cluster_id}, exclude_fields={"status": ClusterStatusEnum.DELETED}
            )
            if not db_cluster:
                raise ClientException("Cluster does not exist")
            request.cluster_id = db_cluster.bud_cluster_id

        # Prepare workflow step data
        workflow_step_data = RunBenchmarkWorkflowStepData(**request.model_dump(exclude={"workflow_id", "workflow_total_steps", "step_number", "trigger_workflow"})).model_dump(mode="json")

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

        # Trigger workflow
        if trigger_workflow:
            # query workflow steps again to get latest data
            db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
                {"workflow_id": db_workflow.id}
            )

            # Define the keys required for model extraction
            keys_of_interest = [
                "name",
                "tags",
                "description",
                "concurrent_requests",
                "eval_with",
                "use_cache",
                "cluster_id",
                "nodes",
                "model_id",
                "user_confirmation",
                "run_as_simulation",
            ]

            keys_of_interest_for_dataset = ["datasets"]
            keys_of_interest_for_configuration = ["max_input_tokens", "max_output_tokens"]
            keys_of_interest_for_cache = ["embedding_model", "eviction_policy", "max_size", "ttl", "score_threshold"]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            # Check if all required keys are present
            required_keys = keys_of_interest
            if required_data.get("eval_with", "") == "dataset":
                required_keys += keys_of_interest_for_dataset
            elif required_data.get("eval_with", "") == "configuration":
                required_keys += keys_of_interest_for_configuration
            if required_data.get("use_cache", False):
                required_keys += keys_of_interest_for_cache
            missing_keys = [key for key in required_keys if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data for run benchmark workflow: {', '.join(missing_keys)}")

            try:
                # Perform add worker deployment
                await self._perform_run_benchmark_request(
                    current_step_number, required_data, db_workflow, current_user_id
                )
            except ClientException as e:
                raise e

        return db_workflow

    async def _add_run_benchmark_workflow_step(self, current_user_id: UUID, request: RunBenchmarkWorkflowRequest, db_workflow: WorkflowModel, current_user_id: UUID):
        """Add run benchmark workflow step."""
        run_benchmark_payload = {
            **request.model_dump(mode="json"),
            "notification_metadata": {
                "name": BUD_INTERNAL_WORKFLOW,
                "subscriber_ids": str(current_user_id),
                "workflow_id": str(db_workflow.id),
            },
            "source_topic": f"{app_settings.source_topic}",
        }

        logger.debug(f"Performing run benchmark request to budcluster {run_benchmark_payload}")

        run_benchmark_response = await self._perform_run_benchmark_request(run_benchmark_payload)

        # Add payload dict to response
        for step in run_benchmark_response["steps"]:
            step["payload"] = {}

        run_benchmark_events = {BudServeWorkflowStepEventName.BUDSERVE_CLUSTER_EVENTS.value: run_benchmark_response}

        current_step_number = current_step_number + 1
        workflow_current_step = current_step_number

        # Update or create next workflow step
        db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
            db_workflow.id, current_step_number, run_benchmark_events
        )
        logger.debug(f"Workflow step created with id {db_workflow_step.id}")

        # Update progress in workflow
        run_benchmark_response["progress_type"] = BudServeWorkflowStepEventName.BUDSERVE_CLUSTER_EVENTS.value
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"progress": run_benchmark_response, "current_step": workflow_current_step}
        )

    @staticmethod
    async def _perform_run_benchmark_request(payload: dict):
        """Perform run benchmark request to budcluster service."""
        run_benchmark_endpoint = (
            f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_cluster_app_id}/method/benchmark/run"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(run_benchmark_endpoint, json=payload) as response:
                    response_data = await response.json()
                    if response.status >= 400 or response_data.get("object") == "error":
                        raise ClientException("Unable to perform run benchmark")
                    return response_data
        except ClientException as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to perform run benchmark request: {e}")
            raise ClientException("Unable to perform run benchmark request to budcluster") from e

    async def update_benchmark_status_from_notification_event(self, payload: NotificationPayload) -> None:
        """Add benchmark from notification event."""
        # Get workflow and workflow steps
        workflow_id = payload.workflow_id
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        # Define the keys required for endpoint creation
        keys_of_interest = [
            "benchmark_id",
        ]

        # from workflow steps extract necessary information
        required_data = {}
        for db_workflow_step in db_workflow_steps:
            for key in keys_of_interest:
                if key in db_workflow_step.data:
                    required_data[key] = db_workflow_step.data[key]

        logger.debug("Collected required data from workflow steps")

        # Get benchmark
        db_benchmark = await BenchmarkCRUD(self.session).fetch_one(
            conditions={"id": required_data["benchmark_id"]},
            raise_on_error=False,
        )

        if not db_benchmark:
            logger.error(f"Benchmark with id {required_data['benchmark_id']} not found")
            return

        benchmark_result = payload.content.result.get("result", {})

        self.session.refresh(db_benchmark)
        db_benchmark = await BenchmarkCRUD(self.session).update(
            data={},
            conditions={"id": db_benchmark.id},
        )
        logger.debug(f"Updated benchmark: {db_benchmark}")

        # Update current step number
        current_step_number = db_workflow.current_step + 1
        workflow_current_step = current_step_number

        execution_status_data = {
            "workflow_execution_status": {
                "status": "success",
                "message": "Benchmark run successfully.",
            },
        }
        # Update or create next workflow step
        db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
            db_workflow.id, current_step_number, execution_status_data
        )
        logger.debug(f"Upsert workflow step {db_workflow_step.id} for storing benchmark run details")

        # Mark workflow as completed
        logger.debug(f"Marking workflow as completed: {workflow_id}")
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"status": WorkflowStatusEnum.COMPLETED, "current_step": workflow_current_step}
        )

        # Send notification to workflow creator
        model_icon = await ModelServiceUtil(self.session).get_model_icon(db_benchmark.model)

        notification_request = (
            NotificationBuilder()
            .set_content(
                title=db_benchmark.name,
                message="Benchmark completed",
                icon=model_icon,
                result=NotificationResult(target_id=db_benchmark.id, target_type="benchmark").model_dump(
                    exclude_none=True, exclude_unset=True
                ),
            )
            .set_payload(workflow_id=str(db_workflow.id), type=NotificationTypeEnum.MODEL_BENCHMARK_SUCCESS.value)
            .set_notification_request(subscriber_ids=[str(db_workflow.created_by)])
            .build()
        )
        await BudNotifyService().send_notification(notification_request)
