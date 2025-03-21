from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import UUID4, BaseModel, ConfigDict, Field, model_validator

from budapp.commons.constants import BenchmarkStatusEnum
from budapp.commons.schemas import PaginatedSuccessResponse

from ..cluster_ops.schemas import ClusterResponse
from ..model_ops.schemas import ModelResponse


class RunBenchmarkWorkflowStepData(BaseModel):
    """Run Benchmark workflow step data."""

    # step 1
    name: str
    tags: list
    description: str
    concurrent_requests: int
    eval_with: Literal["dataset", "configuration"]
    user_id: Optional[UUID] = None

    # step 2
    datasets: Optional[list[UUID]] = None
    # or
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None

    # step 3
    # use_cache: Optional[bool]
    # embedding_model: Optional[str]
    # eviction_policy: Optional[str]
    # max_size: Optional[int]
    # ttl: Optional[int]
    # score_threshold: Optional[float]

    # step 3
    cluster_id: Optional[UUID] = None
    bud_cluster_id: Optional[UUID] = None

    # step 4
    nodes: Optional[list[dict[str, Any]]] = None

    # step 5
    model_id: Optional[UUID] = None
    model: Optional[str] = None
    provider_type: Optional[str] = None

    # step 6
    credential_id: Optional[UUID] = None

    # step 7
    user_confirmation: Optional[bool] = None

    # step 8
    run_as_simulation: Optional[bool] = None


class RunBenchmarkWorkflowRequest(RunBenchmarkWorkflowStepData):
    """Run Benchmark Workflow Request."""
    # workflow metadata
    workflow_id: UUID4 | None = None
    workflow_total_steps: int | None = None
    step_number: int = Field(..., gt=0)
    trigger_workflow: bool = False

    @model_validator(mode="after")
    def validate_fields(self) -> "RunBenchmarkWorkflowRequest":
        """Validate the fields of the request."""
        if self.workflow_id is None and self.workflow_total_steps is None:
            raise ValueError("workflow_total_steps is required when workflow_id is not provided")

        if self.workflow_id is not None and self.workflow_total_steps is not None:
            raise ValueError("workflow_total_steps and workflow_id cannot be provided together")

        # if self.use_cache is True and (self.embedding_model is None or self.eviction_policy is None or self.score_threshold is None):  # noqa: E501self.embedding_model is None:
        #     raise ValueError("embedding_model, eviction_policy and score_threshold must be provided if use_cache is True")

        return self


class BenchmarkFilter(BaseModel):
    name: str | None = None
    model_name: str | None = None
    cluster_name: str | None = None
    min_concurrency: int | None = None
    max_concurrency: int | None = None
    min_tpot: float | None = None
    max_tpot: float | None = None
    min_ttft: float | None = None
    max_ttft: float | None = None


class BenchmarkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID4
    name: str
    status: BenchmarkStatusEnum
    model: ModelResponse
    cluster: ClusterResponse
    node_type: str
    vendor_type: str
    concurrency: int
    tpot: float = 0.5
    ttft: float = 0.5
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def extract_node_info(cls, values):
        """Extract Node type and Vendor type values."""
        if not isinstance(values, dict):  # Ensure values is a dictionary
            values = values.__dict__  # Convert object to dictionary

        nodes = values.get("nodes", [])
        values["node_type"] = ",".join({device["type"] for node in nodes for device in node.get("devices", [])})
        values["vendor_type"] = ",".join({device["name"] for node in nodes for device in node.get("devices", [])})
        return values


class BenchmarkPaginatedResponse(PaginatedSuccessResponse):
    benchmarks: list[BenchmarkResponse]
