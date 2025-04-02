import json
from typing import Dict, List, Optional
from uuid import UUID

import aiohttp
from fastapi import HTTPException, status
from sqlalchemy import text

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin

from ..cluster_ops.crud import ClusterDataManager
from ..cluster_ops.models import Cluster as ClusterModel
from ..cluster_ops.services import ClusterService
from ..commons.config import app_settings
from ..commons.constants import (
    APP_ICONS,
    BUD_INTERNAL_WORKFLOW,
    BenchmarkStatusEnum,
    BudServeWorkflowStepEventName,
    ClusterStatusEnum,
    ModelProviderTypeEnum,
    ModelStatusEnum,
    NotificationTypeEnum,
    WorkflowStatusEnum,
    WorkflowTypeEnum,
)
from ..commons.exceptions import ClientException
from ..core.schemas import NotificationPayload, NotificationResult
from ..credential_ops.crud import ProprietaryCredentialDataManager
from ..credential_ops.models import ProprietaryCredential as ProprietaryCredentialModel
from ..dataset_ops.models import DatasetCRUD
from ..dataset_ops.schemas import DatasetResponse
from ..endpoint_ops.schemas import ModelClusterDetail
from ..endpoint_ops.services import EndpointService
from ..model_ops.crud import ModelDataManager, ProviderDataManager
from ..model_ops.models import Model
from ..model_ops.models import Provider as ProviderModel
from ..model_ops.services import ModelService, ModelServiceUtil
from ..shared.notification_service import BudNotifyService, NotificationBuilder
from ..workflow_ops.crud import WorkflowDataManager, WorkflowStepDataManager
from ..workflow_ops.models import Workflow as WorkflowModel
from ..workflow_ops.models import WorkflowStep as WorkflowStepModel
from ..workflow_ops.schemas import WorkflowUtilCreate
from ..workflow_ops.services import WorkflowService, WorkflowStepService
from .models import BenchmarkCRUD, BenchmarkRequestMetricsCRUD, BenchmarkRequestMetricsSchema, BenchmarkSchema
from .schemas import (
    AddRequestMetricsRequest,
    BenchmarkRequestMetrics,
    RunBenchmarkWorkflowRequest,
    RunBenchmarkWorkflowStepData,
)


logger = logging.get_logger(__name__)

class BenchmarkService(SessionMixin):
    """Benchmark service."""

    async def run_benchmark_workflow(self, current_user_id: UUID, request: RunBenchmarkWorkflowRequest):
        """Run benchmark workflow."""
        # Get request data
        step_number = request.step_number
        workflow_id = request.workflow_id
        workflow_total_steps = request.workflow_total_steps
        trigger_workflow = request.trigger_workflow

        # resources
        model_id = request.model_id
        cluster_id = request.cluster_id
        credential_id = request.credential_id

        # set user_id
        request.user_id = current_user_id

        current_step_number = step_number

        # Retrieve or create workflow
        workflow_create = WorkflowUtilCreate(
            workflow_type=WorkflowTypeEnum.MODEL_BENCHMARK,
            title="Run model benchmark",
            total_steps=workflow_total_steps,
            icon=APP_ICONS["general"]["model_mono"],
            tag="Benchmark",
        )
        db_workflow = await WorkflowService(self.session).retrieve_or_create_workflow(
            workflow_id, workflow_create, current_user_id
        )

        # Get workflow steps
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": db_workflow.id}
        )

        # Validate resources
        if model_id:
            db_model = await ModelDataManager(self.session).retrieve_by_fields(
                Model, fields={"id": model_id}, exclude_fields={"status": ModelStatusEnum.DELETED}
            )
            if not db_model:
                raise ClientException("Model does not exist")
            if db_model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
                model_uri = db_model.uri
                model_source = db_model.source
                if model_uri.startswith(f"{model_source}/"):
                    model_uri = model_uri.removeprefix(f"{model_source}/")
                # Note: When model_id is given, credential_id is not present since credential_id is received in the next step
                # if not credential_id condition will not work, therefore added another section
                # if credential_id and updating request.model value there.
                request.model = model_uri if not credential_id else f"{model_source}/{model_uri}"
                request.provider_type = db_model.provider_type.value
            else:
                request.model = db_model.local_path
                request.provider_type = db_model.provider_type.value
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

        if cluster_id:
            db_cluster = await ClusterDataManager(self.session).retrieve_by_fields(
                ClusterModel, fields={"id": cluster_id}, exclude_fields={"status": ClusterStatusEnum.DELETED}
            )
            if not db_cluster:
                raise ClientException("Cluster does not exist")
            request.bud_cluster_id = db_cluster.cluster_id

        # this section added to update request.model for cloud model providers
        if credential_id:
            db_credential = await ProprietaryCredentialDataManager(self.session).retrieve_by_fields(
                ProprietaryCredentialModel, fields={"id": credential_id}, missing_ok=True
            )
            if not db_credential:
                raise ClientException("Credential does not exist")
            # find model_id in previous step data
            model_id = None
            for step_data in db_workflow_steps:
                model_id = step_data.data.get("model_id")
                if model_id:
                    break
            if model_id:
                db_model = await ModelDataManager(self.session).retrieve_by_fields(
                    Model, fields={"id": model_id}, exclude_fields={"status": ModelStatusEnum.DELETED}
                )
                if not db_model:
                    raise ClientException("Model does not exist")
                if db_model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
                    model_uri = db_model.uri
                    model_source = db_model.source
                    if model_uri.startswith(f"{model_source}/"):
                        model_uri = model_uri.removeprefix(f"{model_source}/")
                    request.model = f"{model_source}/{model_uri}"

        # Prepare workflow step data
        workflow_step_data = RunBenchmarkWorkflowStepData(**request.model_dump(exclude={"workflow_id", "workflow_total_steps", "step_number", "trigger_workflow"})).model_dump(mode="json", exclude_none=True)

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

        # Perform bud simulation
        if model_id:
            db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
                {"workflow_id": db_workflow.id}
            )

            # Define the keys required for model security scan
            keys_of_interest = [
                "model_id",
                "cluster_id",
                "bud_cluster_id",
                "nodes",
                "eval_with",
                "concurrent_requests",
                "max_input_tokens",
                "max_output_tokens",
            ]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            # Check if all required keys are present
            required_keys = ["model_id", "cluster_id", "bud_cluster_id","nodes", "eval_with", "concurrent_requests"]
            if required_data.get("eval_with", "") == "configuration":
                required_keys.extend(["max_input_tokens", "max_output_tokens"])
            missing_keys = [key for key in required_keys if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data for bud simulation: {', '.join(missing_keys)}")
            # Perform add worker bud simulation
            try:
                await self._perform_add_worker_simulation(
                    current_step_number, required_data, db_workflow, current_user_id
                )
            except ClientException as e:
                raise e


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
                "datasets",
                "max_input_tokens",
                "max_output_tokens",
                # "use_cache",
                "cluster_id",
                "bud_cluster_id",
                "nodes",
                "model_id",
                "model",
                "provider_type",
                "credential_id",
                "user_confirmation",
                "run_as_simulation",
                "user_id",
                "simulator_id",
            ]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            # Check if all required keys are present
            required_keys = keys_of_interest
            if required_data.get("eval_with", "") == "dataset":
                required_keys.remove("max_input_tokens")
                required_keys.remove("max_output_tokens")
            elif required_data.get("eval_with", "") == "configuration":
                required_keys.remove("datasets")
            if required_data.get("provider_type", "") in ["hugging_face", "url", "disk"]:
                required_keys.remove("credential_id")

            missing_keys = [key for key in required_keys if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data for run benchmark workflow: {', '.join(missing_keys)}")

            try:
                if "datasets" in required_data:
                    with DatasetCRUD() as crud:
                        db_datasets = await crud.get_datatsets_by_ids(required_data["datasets"])
                        required_data["datasets"] = [DatasetResponse.model_validate(db_dataset).model_dump(mode="json") for db_dataset in db_datasets]
                # Perform add worker deployment
                await self._add_run_benchmark_workflow_step(
                    current_step_number, required_data, db_workflow, current_user_id
                )
            except ClientException as e:
                raise e

        return db_workflow

    async def _add_run_benchmark_workflow_step(self, current_step_number: int, request: dict, db_workflow: WorkflowModel, current_user_id: UUID):
        """Add run benchmark workflow step."""
        # insert benchmark in budapp db
        benchmark_id = None
        with BenchmarkCRUD() as crud:
            db_benchmark =  crud.insert(
                BenchmarkSchema(
                    name=request["name"],
                    tags=request["tags"],
                    description=request["description"],
                    eval_with=request["eval_with"],
                    user_id=current_user_id,
                    model_id=request["model_id"],
                    cluster_id=request["cluster_id"],
                    nodes=request["nodes"],
                    concurrency=request["concurrent_requests"],
                    status=BenchmarkStatusEnum.PROCESSING
                )
            )
            logger.debug(f"Benchmark created with id {db_benchmark.id}")
            benchmark_id = db_benchmark.id

        run_benchmark_payload = {
            "benchmark_id": str(benchmark_id),
            **request,
            "notification_metadata": {
                "name": BUD_INTERNAL_WORKFLOW,
                "subscriber_ids": str(current_user_id),
                "workflow_id": str(db_workflow.id),
            },
            "source_topic": f"{app_settings.source_topic}",
        }

        # Update current workflow step
        db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
            db_workflow.id, current_step_number, run_benchmark_payload
        )

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
        model_icon = APP_ICONS["general"]["model_mono"]
        # Get benchmark
        with BenchmarkCRUD() as crud:
            db_benchmark = crud.fetch_one(
                conditions={"id": required_data["benchmark_id"]},
                raise_on_error=False,
            )

            if not db_benchmark:
                logger.error(f"Benchmark with id {required_data['benchmark_id']} not found")
                return

            benchmark_response = payload.content.result
            logger.info(f"Updating benchmark with response: {benchmark_response}")
            if benchmark_response["benchmark_status"]:
                update_data = {"status": BenchmarkStatusEnum.SUCCESS, "bud_cluster_benchmark_id": benchmark_response["bud_cluster_benchmark_id"], "result": benchmark_response["result"]}
            else:
                update_data = {"status": BenchmarkStatusEnum.FAILED, "reason": benchmark_response["result"]}

            crud.update(
                data=update_data,
                conditions={"id": db_benchmark.id},
            )

            db_benchmark = crud.fetch_one(conditions={"id": db_benchmark.id}, raise_on_error=False)


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
        model_icon = await ModelServiceUtil(self.session).get_model_icon(model_id=db_benchmark.model_id)

        notification_request = (
            NotificationBuilder()
            .set_content(
                title=db_benchmark.name,
                message="Benchmark completed",
                icon=model_icon,
                result=NotificationResult(target_id=db_benchmark.id, target_type="model").model_dump(
                    exclude_none=True, exclude_unset=True
                ),
            )
            .set_payload(workflow_id=str(db_workflow.id), type=NotificationTypeEnum.MODEL_BENCHMARK_SUCCESS.value)
            .set_notification_request(subscriber_ids=[str(db_workflow.created_by)])
            .build()
        )
        await BudNotifyService().send_notification(notification_request)

    async def get_benchmarks(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Optional[Dict] = None,
        order_by: Optional[List] = None,
        search: bool = False,
    ) -> List[dict]:
        """Get all benchmarks."""
        with BenchmarkCRUD() as crud:
            db_benchmarks, total_count = await crud.fetch_many_with_search(filters=filters, order_by=order_by, limit=limit, offset=offset, search=search)
            benchmark_list = []
            for db_benchmark in db_benchmarks:
                benchmark_dict = {**db_benchmark.__dict__}
                benchmark_dict["model"] = {**db_benchmark.model.__dict__}  # Ensure relationships are included
                benchmark_dict["cluster"] = {**db_benchmark.cluster.__dict__}
                benchmark_dict["tpot"] = round(benchmark_dict["result"].get("mean_tpot_ms", 0.0), 2) if benchmark_dict["result"] else 0.0
                benchmark_dict["ttft"] = round(benchmark_dict["result"].get("mean_ttft_ms", 0.0), 2) if benchmark_dict["result"] else 0.0
                benchmark_list.append(benchmark_dict)
            return benchmark_list, total_count

    async def _perform_get_benchmark_result_request(self, benchmark_id: UUID) -> dict:
        """Perform run benchmark request to budcluster service."""
        get_benchmark_result_endpoint = f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_cluster_app_id}/method/benchmark/result"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(get_benchmark_result_endpoint, params={"benchmark_id": str(benchmark_id)}) as response:
                    response_data = await response.json()
                    logger.debug(f"Response from budcluster service: {response_data}")

                    if response.status != 200 or response_data.get("object") == "error":
                        error_message = response_data.get("message", "Failed to get benchmark result")
                        logger.error(
                            f"Failed to get benchmark result with external service: {error_message}"
                        )
                        raise ClientException(error_message, status_code=response.status)

                    logger.debug("Successfully fetched benchmark result with budcluster service")
                    return response_data

            except ClientException as e:
                raise e

            except Exception as e:
                logger.error(f"Failed to make get benchmark result call to budcluster service: {e}")
                raise ClientException("Unable to get benchmark result with external service") from e


    async def get_benchmark_result(self, benchmark_id: UUID) -> dict:
        """Get benchmark result."""
        with BenchmarkCRUD() as crud:  # noqa: SIM117
            with crud.get_session() as session:
                db_benchmark = crud.fetch_one(conditions={"id": benchmark_id}, session=session, raise_on_error=False)

                if not db_benchmark:
                    raise HTTPException(
                        detail=f"Benchmark not found: {benchmark_id}", status_code=status.HTTP_404_NOT_FOUND
                    )
                if db_benchmark.status in [BenchmarkStatusEnum.PROCESSING, BenchmarkStatusEnum.FAILED]:
                    raise HTTPException(
                        detail=f"Benchmark {db_benchmark.name} is {db_benchmark.status}", status_code=status.HTTP_400_BAD_REQUEST
                    )
        bud_cluster_response = await self._perform_get_benchmark_result_request(db_benchmark.bud_cluster_benchmark_id)

        return bud_cluster_response["param"]

    async def get_benchmark_model_cluster_detail(self, benchmark_id: UUID) -> dict:
        """Get benchmark model cluster detail."""
        with BenchmarkCRUD() as crud, crud.get_session() as session:
            db_benchmark = crud.fetch_one(conditions={"id": benchmark_id}, session=session, raise_on_error=False)
            if not db_benchmark:
                raise HTTPException(
                    detail=f"Benchmark not found: {benchmark_id}", status_code=status.HTTP_404_NOT_FOUND
                )
        model_id = db_benchmark.model_id
        model_detail_json_response = await ModelService(self.session).retrieve_model(model_id)
        model_detail = json.loads(model_detail_json_response.body.decode("utf-8"))
        cluster_id = db_benchmark.cluster_id
        cluster_detail = await ClusterService(self.session).get_cluster_details(cluster_id)
        return ModelClusterDetail(
            id=db_benchmark.id,
            name=db_benchmark.name,
            status=db_benchmark.status,
            model=model_detail["model"],
            cluster=cluster_detail,
        )

    def get_field1_vs_field2_data(self, field1: str, field2: str, model_ids: Optional[List[str]] = None) -> dict:
        """Get field1 vs field2 data."""
        GET_DATA_QUERY = f"""
            SELECT
                b.model_id,
                m.uri,
                b.result->>'{field1}' AS {field1},
                b.result->>'{field2}' AS {field2}
            FROM benchmark as b
            JOIN model m ON b.model_id = m.id
            WHERE b.result is not NULL AND b.result->>'{field1}' is not NULL AND b.result->>'{field2}' is not NULL
        """
        if model_ids:
            GET_DATA_QUERY += " AND m.id = ANY(:model_ids)"
        print(GET_DATA_QUERY)
        with BenchmarkCRUD() as crud:
            analysis_data = crud.execute_raw_query(query=text(GET_DATA_QUERY), params={"model_ids": model_ids})

        analysis_data_list = []
        for row in analysis_data:
            analysis_data_list.append({
                "model_id": str(row[0]),
                "model_uri": row[1],
                field1: row[2],
                field2: row[3],
            })
        return analysis_data_list

    async def _perform_add_worker_simulation(
        self, current_step_number: int, data: Dict, db_workflow: WorkflowModel, current_user_id: UUID
    ) -> None:
        """Perform bud simulation."""
        db_model = await ModelDataManager(self.session).retrieve_by_fields(
            Model, fields={"id": data["model_id"]}, exclude_fields={"status": ModelStatusEnum.DELETED}
        )

        # Create request payload
        deployment_config = {
            "max_input_tokens": data.get("max_input_tokens"),
            "max_output_tokens": data.get("max_output_tokens"),
            "concurrent_requests": data["concurrent_requests"],
        }
        payload = {
            "pretrained_model_uri": db_model.uri,
            "input_tokens": deployment_config["max_input_tokens"],
            "output_tokens": deployment_config["max_outut_tokens"],
            "concurrency": deployment_config["concurrent_requests"],
            "cluster_id": str(data["bud_cluster_id"]),
            "nodes": [node["hostname"] for node in data["nodes"]],
            "notification_metadata": {
                "name": BUD_INTERNAL_WORKFLOW,
                "subscriber_ids": str(current_user_id),
                "workflow_id": str(db_workflow.id),
            },
            "source_topic": f"{app_settings.source_topic}",
        }
        if db_model.provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
            payload["is_proprietary_model"] = True
        else:
            payload["is_proprietary_model"] = False

        # Perform bud simulation request
        # bud_simulation_response = await EndpointService(self.session)._perform_bud_simulation_request(payload)
        bud_simulation_response = {
            "eta": 0,
            "steps": [
                {
                    "id": "performance_estimation",
                    "title": "Generating best configuration for each cluster",
                    "description": "Analyze and estimate the optimal performance for each cluster"
                },
                {
                    "id": "ranking",
                    "title": "Ranking the cluster based on performance",
                    "description": "Rank the clusters to find the best configuration"
                }
            ],
            "object": "workflow_metadata",
            "status": "PENDING",
            "workflow_name": "run_simulation",
        }

        # Add payload dict to response
        for step in bud_simulation_response["steps"]:
            step["payload"] = {}

        simulator_id = bud_simulation_response.get("workflow_id")

        # NOTE: Dependency with recommended cluster api (GET /clusters/recommended/{workflow_id})
        # NOTE: Replace concurrent_requests with additional_concurrency
        # Required to compare with concurrent_requests in simulator response
        bud_simulation_events = {
            "simulator_id": simulator_id,
            BudServeWorkflowStepEventName.BUD_SIMULATOR_EVENTS.value: bud_simulation_response,
            "deploy_config": deployment_config,
            "model_id": str(db_model.id),
        }

        # Increment current step number
        current_step_number = current_step_number + 1
        workflow_current_step = current_step_number

        # Update or create next workflow step
        db_workflow_step = await WorkflowStepService(self.session).create_or_update_next_workflow_step(
            db_workflow.id, current_step_number, bud_simulation_events
        )
        logger.debug(f"Workflow step created with id {db_workflow_step.id}")

        # Update progress in workflow
        bud_simulation_response["progress_type"] = BudServeWorkflowStepEventName.BUD_SIMULATOR_EVENTS.value
        await WorkflowDataManager(self.session).update_by_fields(
            db_workflow, {"progress": bud_simulation_response, "current_step": workflow_current_step}
        )


class BenchmarkRequestMetricsService(SessionMixin):
    """Benchmark request metrics service."""

    async def add_request_metrics(self, request: AddRequestMetricsRequest) -> None:
        """Add request metrics."""
        if request.metrics:
            valid_keys = {column.name for column in BenchmarkRequestMetricsSchema.__table__.columns}
            metrics_data = [
                BenchmarkRequestMetricsSchema(**{k: v for k, v in metric.model_dump(mode="json").items() if k in valid_keys})
                for metric in request.metrics
            ]
            with BenchmarkRequestMetricsCRUD() as crud:
                crud.bulk_insert(metrics_data, session=self.session)

    def _get_distribution_bins(self, distribution_type: str, dataset_id: UUID, benchmark_id: Optional[UUID]=None, num_bins: int=10) -> list:
        """Get distribution bins."""
        bins = []
        with BenchmarkRequestMetricsCRUD() as crud:
            params = {"dataset_id": dataset_id}
            query = f"SELECT MAX({distribution_type}) FROM benchmark_request_metrics WHERE dataset_id = :dataset_id"
            if benchmark_id:
                query += " AND benchmark_id = :benchmark_id"
                params["benchmark_id"] = benchmark_id
            metrics_data = crud.execute_raw_query(
                query=text(query),
                params=params,
            )
        if metrics_data:
            max_value = metrics_data[0][0]
            bin_width = max_value / num_bins
            bins = bins = [(i+1, round(i*bin_width, 1), round((i+1)*bin_width, 1)) for i in range(num_bins)]
            # Adjust the bin range to make them exclusive
            bins = [(bin_id, bin_start, bin_end + 0.1 if bin_id == num_bins else bin_end) for bin_id, bin_start, bin_end in bins]
            print(bins)
        return bins

    async def get_dataset_distribution_metrics(self, distribution_type: str, dataset_id: UUID, benchmark_id: Optional[UUID]=None, num_bins: int=10) -> list:
        """Get dataset distribution metrics."""
        graph_data_list = []
        # calculate distribution bins
        bins = self._get_distribution_bins(distribution_type, dataset_id, benchmark_id, num_bins)

        if not bins:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset distribution not found: {dataset_id}",
            )

        with BenchmarkRequestMetricsCRUD() as crud:
            params={"dataset_id": dataset_id}
            query = f"""
                    WITH bins AS (
                        SELECT * FROM (VALUES
                            {', '.join([f"({bin_id}, {bin_start}, {bin_end})" for bin_id, bin_start, bin_end in bins])}
                        ) AS t(bin_id, bin_start, bin_end)
                    )
                    SELECT
                        b.bin_id,
                        b.bin_start || '-' || b.bin_end AS bin_range,
                        ROUND(COALESCE(AVG(m.ttft)::numeric, 0), 2) AS avg_ttft,
                        ROUND(COALESCE(AVG(m.tpot)::numeric, 0), 2) AS avg_tpot,
                        ROUND(COALESCE(AVG(m.latency)::numeric, 0), 2) AS avg_latency,
                        ROUND(COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY m.ttft)::numeric, 0), 2) AS p95_ttft,
                        ROUND(COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY m.tpot)::numeric, 0), 2) AS p95_tpot,
                        ROUND(COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY m.latency)::numeric, 0), 2) AS p95_latency
                    """
            if distribution_type == "prompt_len":
                query += """
                        ,ROUND(COALESCE(AVG(m.output_len)::numeric, 0), 2) AS avg_output_len
                """
            query += f"""
                    FROM bins b
                    LEFT JOIN benchmark_request_metrics m
                        ON m.{distribution_type} >= b.bin_start
                        AND m.{distribution_type} < b.bin_end
                    WHERE m.dataset_id = :dataset_id AND m.success is true
                """
            if benchmark_id:
                query += " AND m.benchmark_id = :benchmark_id"
                params["benchmark_id"] = benchmark_id
            query += """
                    GROUP BY b.bin_id, b.bin_start, b.bin_end
                    ORDER BY b.bin_id;
                """
            print(query)
            graph_data = crud.execute_raw_query(
                query=text(query),
                params={"dataset_id": dataset_id, "benchmark_id": benchmark_id},
            )
            for row in graph_data:
                temp_data = {
                    "bin_id": row[0],
                    "bin_range": row[1],
                    "avg_ttft": float(row[2]),
                    "avg_tpot": float(row[3]),
                    "avg_latency": float(row[4]),
                    "p95_ttft": float(row[5]),
                    "p95_tpot": float(row[6]),
                    "p95_latency": float(row[7]),
                }
                if distribution_type == "prompt_len":
                    temp_data["avg_output_len"] = float(row[8])
                graph_data_list.append(temp_data)
            print(graph_data_list)
        return graph_data_list

    async def get_request_metrics(self, benchmark_id: UUID, offset: int = 0, limit: int = 10) -> dict:
        """Get benchmark request metrics."""
        with BenchmarkRequestMetricsCRUD() as crud:
            db_request_metrics, _ = crud.fetch_many(conditions={"benchmark_id": benchmark_id}, limit=limit, offset=offset)
            total_count = crud.fetch_count(conditions={"benchmark_id": benchmark_id})
            request_metrics = [BenchmarkRequestMetrics.model_validate(request_metric, from_attributes=True) for request_metric in db_request_metrics]
        return request_metrics, total_count
