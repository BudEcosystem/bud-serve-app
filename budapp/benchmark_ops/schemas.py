from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import UUID4, BaseModel, Field, model_validator


class RunBenchmarkWorkflowStepData(BaseModel):
    """Run Benchmark workflow step data."""

    # step 1
    name: str
    tags: list
    description: str
    concurrent_requests: int
    eval_with: Literal["dataset", "configuration"]

    # step 2
    datasets: Optional[list[UUID]]
    # or
    max_input_tokens: Optional[int]
    max_output_tokens: Optional[int]

    # step 3
    # use_cache: Optional[bool]
    # embedding_model: Optional[str]
    # eviction_policy: Optional[str]
    # max_size: Optional[int]
    # ttl: Optional[int]
    # score_threshold: Optional[float]

    # step 4
    cluster_id: Optional[UUID]

    # step 5
    nodes: Optional[list[dict[str, Any]]]

    # step 6
    model_id: Optional[UUID]
    model: Optional[str]

    # step 7
    credential_id: Optional[UUID]

    # step 8
    user_confirmation: Optional[bool]

    # step 9
    run_as_simulation: Optional[bool]

    @model_validator(mode="after")
    def validate_fields(self) -> "RunBenchmarkWorkflowStepData":
        """Validate the fields of the request."""
        if self.datasets is None and (self.max_input_tokens is None or self.max_output_tokens is None):
            raise ValueError("At least one of datasets or configuration (max_input_tokens and max_output_tokens) is required")
        # if self.use_cache is True and (self.embedding_model is None or self.eviction_policy is None or self.score_threshold is None):  # noqa: E501self.embedding_model is None:
        #     raise ValueError("embedding_model, eviction_policy and score_threshold must be provided if use_cache is True")
        return self


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

        return self
