from typing import Dict, List, Tuple
from uuid import UUID

from fastapi import status

from budapp.commons import logging
from budapp.commons.constants import (
    WorkflowStatusEnum,
)
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.core.crud import WorkflowDataManager, WorkflowStepDataManager
from budapp.core.models import Workflow as WorkflowModel
from budapp.core.models import WorkflowStep as WorkflowStepModel

from .crud import CloudModelDataManager, ModelDataManager, ProviderDataManager
from .models import CloudModel, Model
from .models import Provider as ProviderModel
from .schemas import (
    CreateCloudModelWorkflowRequest,
    CreateCloudModelWorkflowResponse,
    CreateCloudModelWorkflowStepData,
    CreateCloudModelWorkflowSteps,
)


logger = logging.get_logger(__name__)


class ProviderService(SessionMixin):
    """Provider service."""

    async def get_all_providers(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[ProviderModel], int]:
        """Get all providers."""
        return await ProviderDataManager(self.session).get_all_providers(offset, limit, filters, order_by, search)


class ModelService(SessionMixin):
    """Model service."""

    async def add_cloud_model_workflow(self, current_user_id: UUID, request: CreateCloudModelWorkflowRequest) -> None:
        """Add a cloud model workflow."""
        # Get request data
        step_number = request.step_number
        workflow_id = request.workflow_id
        workflow_total_steps = request.workflow_total_steps
        provider_type = request.provider_type
        name = request.name
        modality = request.modality
        uri = request.uri
        tags = request.tags
        icon = request.icon
        trigger_workflow = request.trigger_workflow
        provider_id = request.provider_id
        cloud_model_id = request.cloud_model_id
        description = request.description

        current_step_number = step_number

        # Validate workflow
        if workflow_id:
            db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(
                WorkflowModel, {"id": workflow_id}
            )

            if db_workflow.status != WorkflowStatusEnum.IN_PROGRESS:
                logger.error(f"Workflow {workflow_id} is not in progress")
                raise ClientException("Workflow is not in progress")

            if db_workflow.created_by != current_user_id:
                logger.error(f"User {current_user_id} is not the creator of workflow {workflow_id}")
                raise ClientException("User is not authorized to perform this action")
        elif workflow_total_steps:
            db_workflow = await WorkflowDataManager(self.session).insert_one(
                WorkflowModel(total_steps=workflow_total_steps, created_by=current_user_id),
            )
        else:
            raise ClientException("Either workflow_id or workflow_total_steps should be provided")

        # Model source is provider type
        source = None
        if provider_id:
            db_provider = await ProviderDataManager(self.session).retrieve_by_fields(
                ProviderModel, {"id": provider_id}
            )
            source = db_provider.type

        # Prepare workflow step data
        workflow_step_data = CreateCloudModelWorkflowSteps(
            provider_type=provider_type,
            source=source.value if source else None,
            name=name,
            modality=modality,
            uri=uri,
            tags=tags,
            icon=icon,
            provider_id=provider_id,
            cloud_model_id=cloud_model_id,
            description=description,
        ).model_dump(exclude_none=True, exclude_unset=True, mode="json")

        # Get workflow steps
        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": db_workflow.id}
        )

        # For avoiding another db call for record retrieval, storing db object while iterating over db_workflow_steps
        db_current_workflow_step = None

        if db_workflow_steps:
            db_step_uri = None
            db_step_source = None

            for db_step in db_workflow_steps:
                # Get current workflow step
                if db_step.step_number == current_step_number:
                    db_current_workflow_step = db_step

                if "uri" in db_step.data:
                    db_step_uri = db_step.data["uri"]
                if "source" in db_step.data:
                    db_step_source = db_step.data["source"]

            # Check duplicate endpoint in project
            query_uri = None
            query_source = None
            if uri and db_step_source:
                # If user gives uri but source given in earlier step
                query_uri = uri
                query_source = db_step_source
            elif source and db_step_uri:
                # If user gives source but uri given in earlier step
                query_uri = db_step_uri
                query_source = source.value
            elif uri and source:
                # if user gives both source and uri
                query_uri = uri
                query_source = source.value
            elif db_step_uri and db_step_source:
                # if user gives source and uri in earlier step
                query_uri = db_step_uri
                query_source = db_step_source

            if query_uri and query_source:
                # NOTE: A model can only be deployed once in a project
                db_cloud_model = await ModelDataManager(self.session).retrieve_by_fields(
                    Model,
                    {
                        "uri": query_uri,
                        "source": query_source,
                        "is_active": True,
                    },
                    missing_ok=True,
                )
                if db_cloud_model:
                    logger.info(f"Model {query_uri} already added with {query_source}")
                    raise ClientException("Duplicate model uri and source found")

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

            # Default values are inserted in first step of a workflow
            if not db_workflow_steps:
                workflow_step_data["created_by"] = str(current_user_id)

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

        # Update workflow step data in db
        db_workflow = await WorkflowDataManager(self.session).update_by_fields(
            db_workflow,
            {"current_step": workflow_current_step},
        )

        # Execute workflow
        if trigger_workflow:
            logger.info("Workflow triggered")

            # TODO: Currently querying workflow steps again by ordering steps in ascending order
            # To ensure the latest step update is fetched, Consider excluding it later
            db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
                {"workflow_id": db_workflow.id}
            )

            # Define the keys required for model deployment
            keys_of_interest = [
                "source",
                "name",
                "modality",
                "uri",
                "tags",
                "icon",
                "provider_type",
                "provider_id",
                "description",
            ]

            # from workflow steps extract necessary information
            required_data = {}
            for db_workflow_step in db_workflow_steps:
                for key in keys_of_interest:
                    if key in db_workflow_step.data:
                        required_data[key] = db_workflow_step.data[key]

            # Check if all required keys are present
            missing_keys = [key for key in keys_of_interest if key not in required_data]
            if missing_keys:
                raise ClientException(f"Missing required data: {', '.join(missing_keys)}")

            # Trigger deploy model by step
            db_model = await ModelDataManager(self.session).insert_one(Model(**required_data))
            logger.info("Successfully triggered model deployment")

        return db_workflow

    async def get_cloud_model_workflow(self, workflow_id: UUID) -> CreateCloudModelWorkflowResponse:
        """Get cloud model workflow."""
        db_workflow = await WorkflowDataManager(self.session).retrieve_by_fields(WorkflowModel, {"id": workflow_id})

        db_workflow_steps = await WorkflowStepDataManager(self.session).get_all_workflow_steps(
            {"workflow_id": workflow_id}
        )

        if not db_workflow_steps:
            return CreateCloudModelWorkflowResponse(
                workflow_id=db_workflow.id,
                status=db_workflow.status,
                current_step=db_workflow.current_step,
                total_steps=db_workflow.total_steps,
                reason=db_workflow.reason,
                workflow_steps=CreateCloudModelWorkflowStepData(),
                code=status.HTTP_200_OK,
            )

        # Define the keys required for model deployment
        keys_of_interest = [
            "source",
            "name",
            "modality",
            "uri",
            "tags",
            "icon",
            "provider_type",
            "provider_id",
            "cloud_model_id",
            "description",
        ]

        # from workflow steps extract necessary information
        required_data = {}
        for db_workflow_step in db_workflow_steps:
            for key in keys_of_interest:
                if key in db_workflow_step.data:
                    required_data[key] = db_workflow_step.data[key]

        provider_type = required_data.get("provider_type")

        db_provider = (
            await ProviderDataManager(self.session).retrieve_by_fields(
                ProviderModel, {"id": required_data["provider_id"]}, missing_ok=True
            )
            if "provider_id" in required_data
            else None
        )

        db_cloud_model = (
            await CloudModelDataManager(self.session).retrieve_by_fields(
                CloudModel, {"id": required_data["cloud_model_id"]}, missing_ok=True
            )
            if "cloud_model_id" in required_data
            else None
        )

        # budserve_cluster_events = required_data.get("budserve_cluster_events")
        # bud_simulator_events = required_data.get("bud_simulator_events")
        # simulator_id = required_data.get("simulator_id")
        # template_id = required_data.get("template_id")
        # deploy_config = required_data.get("deploy_config")

        # return ModelDeployWorkflowResponse(
        #     workflow_id=db_workflow.id,
        #     status=db_workflow.status,
        #     current_step=db_workflow.current_step,
        #     total_steps=db_workflow.total_steps,
        #     reason=db_workflow.reason,
        #     workflow_steps=ModelDeployWorkflowStepResponse(
        #         model=db_model,
        #         cluster=db_cluster,
        #         project=db_project,
        #         template=db_template,
        #         endpoint_name=endpoint_name,
        #         budserve_cluster_events=budserve_cluster_events,
        #         bud_simulator_events=bud_simulator_events,
        #         simulator_id=simulator_id,
        #         template_id=template_id,
        #         deploy_config=deploy_config,
        #     ),
        # )

        return CreateCloudModelWorkflowResponse(
            workflow_id=db_workflow.id,
            status=db_workflow.status,
            current_step=db_workflow.current_step,
            total_steps=db_workflow.total_steps,
            reason=db_workflow.reason,
            workflow_steps=CreateCloudModelWorkflowStepData(
                provider_type=provider_type,
                provider=db_provider,
                cloud_model=db_cloud_model,
            ),
            code=status.HTTP_200_OK,
        )


class CloudModelService(SessionMixin):
    """Cloud model service."""

    async def get_all_cloud_models(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[CloudModel], int]:
        """Get all cloud models."""
        return await CloudModelDataManager(self.session).get_all_cloud_models(offset, limit, filters, order_by, search)

    async def get_all_recommended_tags(
        self,
        offset: int = 0,
        limit: int = 10,
    ) -> Tuple[List[CloudModel], int]:
        """Get all cloud models."""
        return await CloudModelDataManager(self.session).get_all_recommended_tags(offset, limit)
