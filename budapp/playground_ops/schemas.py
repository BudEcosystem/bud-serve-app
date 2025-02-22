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


"""Contains core Pydantic schemas used for data validation and serialization within the model ops services."""

import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from ..commons.schemas import PaginatedSuccessResponse
from ..endpoint_ops.schemas import EndpointListResponse


class PlaygroundDeploymentListResponse(PaginatedSuccessResponse):
    """Playground deployment list response schema."""

    endpoints: list[EndpointListResponse] = []


class PlaygroundDeploymentFilter(BaseModel):
    """Playground deployment filter schema."""

    model_config = ConfigDict(protected_namespaces=())

    name: str | None = None
    model_name: str | None = None
    model_size: str | None = None

    @field_validator("model_size")
    def parse_model_size(cls, v: Optional[str]) -> Optional[int]:
        """Convert the model size string to a number in billions."""
        if not v:
            return None

        try:
            # Match only if string starts with digits and contains only digits
            match = re.match(r"^\d+$", v.strip())
            if match:
                return int(match.group())
            return None
        except Exception:
            return None
