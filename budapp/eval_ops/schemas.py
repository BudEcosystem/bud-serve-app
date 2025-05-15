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

"""The eval ops schemas. Contains schemas for eval ops."""

from typing import List, Optional

from pydantic import UUID4, BaseModel, Field

from budapp.commons.schemas import SuccessResponse


class CreateEvaluationRequest(BaseModel):
    """The request to create an evaluation."""

    name: str = Field(..., description="The name of the evaluation.")
    description: Optional[str] = Field(None, description="The description of the evaluation.")
    project_id: str = Field(..., description="The project ID for the evaluation.")


class Evaluation(BaseModel):
    """The evaluation."""

    evaluation_id: UUID4 = Field(..., description="The ID of the evaluation.")
    name: str = Field(..., description="The name of the evaluation.")
    description: str = Field(..., description="The description of the evaluation.")
    project_id: str = Field(..., description="The project ID for the evaluation.")

class CreateEvaluationResponse(SuccessResponse):
    """The response to create an evaluation."""

    evaluation: Evaluation = Field(..., description="The evaluation.")


class ListEvaluationsResponse(SuccessResponse):
    """The response to list evaluations."""

    evaluations: List[Evaluation] = Field(..., description="The evaluations.")
    
# Update Evaluation Request
class UpdateEvaluationRequest(BaseModel):
    """Request schema to update an evaluation."""
    name: Optional[str] = Field(None, description="The name of the evaluation.")
    description: Optional[str] = Field(None, description="The description of the evaluation.")

class UpdateEvaluationResponse(SuccessResponse):
    """Response schema for updating an evaluation."""
    evaluation: Evaluation = Field(..., description="The updated evaluation.")
    
# Delete Evaluation Request
class DeleteEvaluationResponse(SuccessResponse):
    """Response schema for deleting an evaluation."""
    pass