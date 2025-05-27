# budapp/eval_ops/schemas.py

from typing import List, Optional

from pydantic import UUID4, BaseModel, Field

from budapp.commons.schemas import SuccessResponse
from budapp.eval_ops.models import EvaluationRunStatusEnum


class CreateExperimentRequest(BaseModel):
    """The request to create an experiment."""
    name: str = Field(..., description="The name of the experiment.")
    description: Optional[str] = Field(None, description="The description of the experiment.")
    project_id: UUID4 = Field(..., description="The project ID for the experiment.")


class Experiment(BaseModel):
    """Represents an experiment record."""
    id: UUID4 = Field(..., description="The UUID of the experiment.")
    name: str = Field(..., description="The name of the experiment.")
    description: Optional[str] = Field(None, description="The description of the experiment.")
    project_id: UUID4 = Field(..., description="The project ID for the experiment.")

    class Config:  # noqa: D106
        from_attributes = True


class CreateExperimentResponse(SuccessResponse):
    """The response to create an experiment."""
    experiment: Experiment = Field(..., description="The created experiment.")


class ListExperimentsResponse(SuccessResponse):
    """The response to list experiments."""
    experiments: List[Experiment] = Field(..., description="The experiments.")


class UpdateExperimentRequest(BaseModel):
    """Request schema to update an experiment."""
    name: Optional[str] = Field(None, description="The name of the experiment.")
    description: Optional[str] = Field(None, description="The description of the experiment.")


class UpdateExperimentResponse(SuccessResponse):
    """Response schema for updating an experiment."""
    experiment: Experiment = Field(..., description="The updated experiment.")


class DeleteExperimentResponse(SuccessResponse):
    """Response schema for deleting an experiment."""
    pass


class DatasetBasic(BaseModel):
    """Basic dataset information for trait responses."""
    id: UUID4 = Field(..., description="The UUID of the dataset.")
    name: str = Field(..., description="The name of the dataset.")
    description: Optional[str] = Field(None, description="The description of the dataset.")
    estimated_input_tokens: Optional[int] = Field(None, description="Estimated input tokens.")
    estimated_output_tokens: Optional[int] = Field(None, description="Estimated output tokens.")
    modalities: Optional[List[str]] = Field(None, description="List of modalities.")
    sample_questions_answers: Optional[dict] = Field(None, description="Sample questions and answers in JSON format.")
    advantages_disadvantages: Optional[dict] = Field(None, description="Advantages and disadvantages with structure {'advantages': ['str1'], 'disadvantages': ['str2']}.")

    class Config:  # noqa: D106
        from_attributes = True


class Trait(BaseModel):
    """A trait that experiments can be grouped by."""
    id: UUID4 = Field(..., description="The UUID of the trait.")
    name: str = Field(..., description="The name of the trait.")
    description: Optional[str] = Field(None, description="The description of the trait.")
    # if you still need these for your UI you can keep them:
    category: Optional[str] = Field(None, description="Optional category metadata.")
    exps_ids: List[UUID4] = Field([], description="Optional list of experiment UUIDs.")
    datasets: List[DatasetBasic] = Field([], description="List of datasets associated with this trait.")

    class Config:  # noqa: D106
        from_attributes = True


class ListTraitsResponse(SuccessResponse):
    """The response schema for listing traits."""
    traits: List[Trait] = Field(..., description="The traits.")
    total_record: int = Field(..., description="Total number of traits matching the query.")
    page: int = Field(..., description="Current page number.")
    limit: int = Field(..., description="Number of traits per page.")


class ExpDataset(BaseModel):
    """Represents an evaluation dataset with traits."""
    id: UUID4 = Field(..., description="The UUID of the dataset.")
    name: str = Field(..., description="The name of the dataset.")
    description: Optional[str] = Field(None, description="The description of the dataset.")
    meta_links: Optional[dict] = Field(None, description="Links to GitHub, paper, website, etc.")
    config_validation_schema: Optional[dict] = Field(None, description="Configuration validation schema.")
    estimated_input_tokens: Optional[int] = Field(None, description="Estimated input tokens.")
    estimated_output_tokens: Optional[int] = Field(None, description="Estimated output tokens.")
    language: Optional[List[str]] = Field(None, description="Languages supported by the dataset.")
    domains: Optional[List[str]] = Field(None, description="Domains covered by the dataset.")
    concepts: Optional[List[str]] = Field(None, description="Concepts covered by the dataset.")
    humans_vs_llm_qualifications: Optional[List[str]] = Field(None, description="Human vs LLM qualifications.")
    task_type: Optional[List[str]] = Field(None, description="Types of tasks in the dataset.")
    modalities: Optional[List[str]] = Field(
        None,
        description="List of modalities. Allowed values: 'text' (Textual data), 'image' (Image data), 'video' (Video data)"
    )
    sample_questions_answers: Optional[dict] = Field(None, description="Sample questions and answers in JSON format.")
    advantages_disadvantages: Optional[dict] = Field(None, description="Advantages and disadvantages with structure {'advantages': ['str1'], 'disadvantages': ['str2']}.")
    traits: List[Trait] = Field([], description="Traits associated with this dataset.")

    class Config:  # noqa: D106
        from_attributes = True


class GetDatasetResponse(SuccessResponse):
    """Response schema for getting a dataset by ID."""
    dataset: ExpDataset = Field(..., description="The dataset with traits information.")


class ListDatasetsResponse(SuccessResponse):
    """Response schema for listing datasets."""
    datasets: List[ExpDataset] = Field(..., description="List of datasets with traits.")
    total_record: int = Field(..., description="Total number of datasets matching the query.")
    page: int = Field(..., description="Current page number.")
    limit: int = Field(..., description="Number of datasets per page.")


class DatasetFilter(BaseModel):
    """Filter parameters for dataset listing."""
    name: Optional[str] = Field(None, description="Filter by dataset name (case-insensitive substring).")
    modalities: Optional[List[str]] = Field(None, description="Filter by modalities.")
    language: Optional[List[str]] = Field(None, description="Filter by languages.")
    domains: Optional[List[str]] = Field(None, description="Filter by domains.")


class CreateDatasetRequest(BaseModel):
    """Request schema for creating a new dataset."""
    name: str = Field(..., description="The name of the dataset.")
    description: Optional[str] = Field(None, description="The description of the dataset.")
    meta_links: Optional[dict] = Field(None, description="Links to GitHub, paper, website, etc.")
    config_validation_schema: Optional[dict] = Field(None, description="Configuration validation schema.")
    estimated_input_tokens: Optional[int] = Field(None, description="Estimated input tokens.")
    estimated_output_tokens: Optional[int] = Field(None, description="Estimated output tokens.")
    language: Optional[List[str]] = Field(None, description="Languages supported by the dataset.")
    domains: Optional[List[str]] = Field(None, description="Domains covered by the dataset.")
    concepts: Optional[List[str]] = Field(None, description="Concepts covered by the dataset.")
    humans_vs_llm_qualifications: Optional[List[str]] = Field(None, description="Human vs LLM qualifications.")
    task_type: Optional[List[str]] = Field(None, description="Types of tasks in the dataset.")
    modalities: Optional[List[str]] = Field(
        None,
        description="List of modalities. Allowed values: 'text' (Textual data), 'image' (Image data), 'video' (Video data)"
    )
    sample_questions_answers: Optional[dict] = Field(None, description="Sample questions and answers in JSON format.")
    advantages_disadvantages: Optional[dict] = Field(None, description="Advantages and disadvantages with structure {'advantages': ['str1'], 'disadvantages': ['str2']}.")
    trait_ids: Optional[List[UUID4]] = Field([], description="List of trait IDs to associate with the dataset.")


class UpdateDatasetRequest(BaseModel):
    """Request schema for updating a dataset."""
    name: Optional[str] = Field(None, description="The name of the dataset.")
    description: Optional[str] = Field(None, description="The description of the dataset.")
    meta_links: Optional[dict] = Field(None, description="Links to GitHub, paper, website, etc.")
    config_validation_schema: Optional[dict] = Field(None, description="Configuration validation schema.")
    estimated_input_tokens: Optional[int] = Field(None, description="Estimated input tokens.")
    estimated_output_tokens: Optional[int] = Field(None, description="Estimated output tokens.")
    language: Optional[List[str]] = Field(None, description="Languages supported by the dataset.")
    domains: Optional[List[str]] = Field(None, description="Domains covered by the dataset.")
    concepts: Optional[List[str]] = Field(None, description="Concepts covered by the dataset.")
    humans_vs_llm_qualifications: Optional[List[str]] = Field(None, description="Human vs LLM qualifications.")
    task_type: Optional[List[str]] = Field(None, description="Types of tasks in the dataset.")
    modalities: Optional[List[str]] = Field(
        None,
        description="List of modalities. Allowed values: 'text' (Textual data), 'image' (Image data), 'video' (Video data)"
    )
    sample_questions_answers: Optional[dict] = Field(None, description="Sample questions and answers in JSON format.")
    advantages_disadvantages: Optional[dict] = Field(None, description="Advantages and disadvantages with structure {'advantages': ['str1'], 'disadvantages': ['str2']}.")
    trait_ids: Optional[List[UUID4]] = Field(None, description="List of trait IDs to associate with the dataset.")


class CreateDatasetResponse(SuccessResponse):
    """Response schema for creating a dataset."""
    dataset: ExpDataset = Field(..., description="The created dataset with traits information.")


class UpdateDatasetResponse(SuccessResponse):
    """Response schema for updating a dataset."""
    dataset: ExpDataset = Field(..., description="The updated dataset with traits information.")


class DeleteDatasetResponse(SuccessResponse):
    """Response schema for deleting a dataset."""
    pass

# --- Run schemas ---

class Run(BaseModel):
    """Represents a run of an experiment."""
    id: UUID4 = Field(..., description="UUID of the run.")
    status: EvaluationRunStatusEnum = Field(..., description="Current status of the run.")
    experiment_id: UUID4 = Field(..., description="UUID of the parent experiment.")

    class Config:  # noqa: D106
        from_attributes = True

#-- Run Creation

class RunDatasetConfig(BaseModel):
    """The configuration for a run dataset."""
    dataset_id: UUID4 = Field(..., description="UUID of the dataset.")
    dataset_config: dict = Field(..., description="The configuration validation schema.") # For Each Dataset should have config

class RunTratsConfig(BaseModel):
    """The configuration for a run trait."""
    trait_ids: List[UUID4] = Field(..., description="UUID of the traits.")
    dataset_configs: List[RunDatasetConfig] = Field(..., description="The configuration for each dataset.")

class CreateRunRequest(BaseModel):
    """Payload to create a new run under an experiment."""
    id: Optional[UUID4] = Field(None, description="UUID of the run.") # Multi Step Workflow
    experiment_id: UUID4 = Field(..., description="UUID of the parent experiment.")
    run_traits_config: RunTratsConfig = Field(..., description="The configuration for the run traits.")
    exp_model_config: dict = Field(..., description="The configuration validation schema.") # For Each Model should have config | Update the schema as needed

    class Config:
        """Config for the CreateRunRequest schema."""
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "experiment_id": "123e4567-e89b-12d3-a456-426614174001",
                "run_traits_config": {
                    "trait_ids": [
                        "123e4567-e89b-12d3-a456-426614174002",
                        "123e4567-e89b-12d3-a456-426614174003"
                    ],
                    "dataset_configs": [
                        {
                            "dataset_id": "123e4567-e89b-12d3-a456-426614174004",
                            "dataset_config": {
                                "param1": "value1",
                                "param2": 42
                            }
                        }
                    ]
                },
                "model_config": {
                    "model_param1": "foo",
                    "model_param2": 123
                }
            }
        }


class CreateRunResponse(SuccessResponse):
    """Response after creating a run."""
    run: Run = Field(..., description="The created run.")


class ListRunsResponse(SuccessResponse):
    """Response schema for listing runs."""
    runs: List[Run] = Field(..., description="List of runs.")


class UpdateRunRequest(BaseModel):
    """Payload to update an existing run."""
    status: Optional[EvaluationRunStatusEnum] = Field(None, description="New status of the run.")


class UpdateRunResponse(SuccessResponse):
    """Response after updating a run."""
    run: Run = Field(..., description="The updated run.")


class DeleteRunResponse(SuccessResponse):
    """Response after deleting (soft) a run."""
    pass