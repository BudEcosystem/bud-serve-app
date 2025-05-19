# budapp/eval_ops/schemas.py

from typing import List, Optional

from pydantic import UUID4, BaseModel, Field

from budapp.commons.schemas import SuccessResponse
from budapp.eval_ops.models import EvaluationRunStatusEnum


class CreateEvaluationRequest(BaseModel):
    """The request to create an evaluation."""
    name: str = Field(..., description="The name of the evaluation.")
    description: Optional[str] = Field(None, description="The description of the evaluation.")
    project_id: UUID4 = Field(..., description="The project ID for the evaluation.")


class Evaluation(BaseModel):
    """Represents an evaluation record."""
    id: UUID4 = Field(..., description="The UUID of the evaluation.")
    name: str = Field(..., description="The name of the evaluation.")
    description: Optional[str] = Field(None, description="The description of the evaluation.")
    project_id: UUID4 = Field(..., description="The project ID for the evaluation.")

    class Config:  # noqa: D106
        orm_mode = True


class CreateEvaluationResponse(SuccessResponse):
    """The response to create an evaluation."""
    evaluation: Evaluation = Field(..., description="The created evaluation.")


class ListEvaluationsResponse(SuccessResponse):
    """The response to list evaluations."""
    evaluations: List[Evaluation] = Field(..., description="The evaluations.")


class UpdateEvaluationRequest(BaseModel):
    """Request schema to update an evaluation."""
    name: Optional[str] = Field(None, description="The name of the evaluation.")
    description: Optional[str] = Field(None, description="The description of the evaluation.")


class UpdateEvaluationResponse(SuccessResponse):
    """Response schema for updating an evaluation."""
    evaluation: Evaluation = Field(..., description="The updated evaluation.")


class DeleteEvaluationResponse(SuccessResponse):
    """Response schema for deleting an evaluation."""
    pass


class Trait(BaseModel):
    """A trait that evaluations can be grouped by."""
    id: UUID4 = Field(..., description="The UUID of the trait.")
    name: str = Field(..., description="The name of the trait.")
    description: Optional[str] = Field(None, description="The description of the trait.")
    # if you still need these for your UI you can keep them:
    category: Optional[str] = Field(None, description="Optional category metadata.")
    evals_ids: List[UUID4] = Field([], description="Optional list of evaluation UUIDs.")

    class Config:  # noqa: D106
        orm_mode = True


class ListTraitsResponse(SuccessResponse):
    """The response schema for listing traits."""
    traits: List[Trait] = Field(..., description="The traits.")
    total_record: int = Field(..., description="Total number of traits matching the query.")
    page: int = Field(..., description="Current page number.")
    limit: int = Field(..., description="Number of traits per page.")

# --- Run schemas ---

class Run(BaseModel):
    """Represents a run of an evaluation."""
    id: UUID4 = Field(..., description="UUID of the run.")
    name: str = Field(..., description="Name of the run.")
    description: Optional[str] = Field(None, description="Optional description of the run.")
    status: EvaluationRunStatusEnum = Field(..., description="Current status of the run.")
    evaluation_id: UUID4 = Field(..., description="UUID of the parent evaluation.")

    class Config:  # noqa: D106
        orm_mode = True


class CreateRunRequest(BaseModel):
    """Payload to create a new run under an evaluation."""
    name: str = Field(..., description="Name of the run.")
    description: Optional[str] = Field(None, description="Optional description of the run.")


class CreateRunResponse(SuccessResponse):
    """Response after creating a run."""
    run: Run = Field(..., description="The created run.")


class ListRunsResponse(SuccessResponse):
    """Response schema for listing runs."""
    runs: List[Run] = Field(..., description="List of runs.")


class UpdateRunRequest(BaseModel):
    """Payload to update an existing run."""
    name: Optional[str] = Field(None, description="New name of the run.")
    description: Optional[str] = Field(None, description="New description of the run.")
    status: Optional[EvaluationRunStatusEnum] = Field(None, description="New status of the run.")


class UpdateRunResponse(SuccessResponse):
    """Response after updating a run."""
    run: Run = Field(..., description="The updated run.")


class DeleteRunResponse(SuccessResponse):
    """Response after deleting (soft) a run."""
    pass
