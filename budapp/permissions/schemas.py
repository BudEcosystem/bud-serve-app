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

"""The schemas package, containing the schemas for the permissions."""

import json
from uuid import UUID

from pydantic import BaseModel, field_validator


class ProjectPermissionCreate(BaseModel):
    project_id: UUID
    user_id: UUID
    auth_id: UUID
    scopes: str

    @field_validator("scopes", mode="before")
    @classmethod
    def convert_string_to_json(cls, v):
        if isinstance(v, list):
            return json.dumps(v)
        elif isinstance(v, str):
            return v
        raise ValueError(v)


class PermissionList(BaseModel):
    name: str
    has_permission: bool


class PermissionCreate(BaseModel):
    user_id: UUID
    auth_id: UUID
    scopes: str

    @field_validator("scopes", mode="before")
    @classmethod
    def convert_string_to_json(cls, v):
        if isinstance(v, list):
            return json.dumps(v)
        elif isinstance(v, str):
            return v
        raise ValueError(v)
