from pydantic import UUID4, BaseModel, ConfigDict, Field

from budapp.commons.constants import ModelProviderTypeEnum, WorkflowStatusEnum
from budapp.commons.schemas import SuccessResponse, Tag
from budapp.model_ops.schemas import CloudModel, Model, ModelSecurityScanResult, Provider


class RetrieveWorkflowStepData(BaseModel):
    """Workflow step data schema."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    provider_type: ModelProviderTypeEnum | None = None
    provider: Provider | None = None
    cloud_model: CloudModel | None = None
    cloud_model_id: UUID4 | None = None
    provider_id: UUID4 | None = None
    model_id: UUID4 | None = None
    model: Model | None = None
    workflow_execution_status: dict | None = None
    leaderboard: list | dict | None = None
    name: str | None = None
    ingress_url: str | None = None
    create_cluster_events: dict | None = None
    delete_cluster_events: dict | None = None
    delete_endpoint_events: dict | None = None
    model_security_scan_events: dict | None = None
    icon: str | None = None
    uri: str | None = None
    author: str | None = None
    tags: list[Tag] | None = None
    model_extraction_events: dict | None = None
    description: str | None = None
    security_scan_result_id: UUID4 | None = None
    security_scan_result: ModelSecurityScanResult | None = None


class RetrieveWorkflowDataResponse(SuccessResponse):
    """Retrieve Workflow Data Response."""

    workflow_id: UUID4
    status: WorkflowStatusEnum
    current_step: int
    total_steps: int
    reason: str | None = None
    workflow_steps: RetrieveWorkflowStepData | None = None


class WorkflowResponse(SuccessResponse):
    """Workflow response schema."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    id: UUID4 = Field(alias="workflow_id")
    total_steps: int = Field(..., gt=0)
    status: WorkflowStatusEnum
    current_step: int
    reason: str | None = None
