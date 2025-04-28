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

"""The schemas package, containing the schemas for the user ops."""

from datetime import datetime

from pydantic import UUID4, BaseModel, ConfigDict, EmailStr, Field, field_validator
from typing import Dict, List, Optional, Union
from ..commons.helpers import validate_password_string
from ..permissions.schemas import PermissionList

from budapp.commons.constants import UserRoleEnum, UserStatusEnum
from budapp.commons.schemas import SuccessResponse, PaginatedSuccessResponse


class UserBase(BaseModel):
    """Base user schema."""

    name: str = Field(min_length=1, max_length=100)
    email: EmailStr = Field(min_length=1, max_length=100)


class UserInfo(UserBase):
    """User response to client schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID4
    color: str
    role: UserRoleEnum
    status: UserStatusEnum


class User(UserInfo):
    """User schema."""

    model_config = ConfigDict(from_attributes=True)

    auth_id: UUID4
    password: str
    created_at: datetime
    modified_at: datetime
    raw_token: str | None = None


class UserResponse(SuccessResponse):
    """User response to client schema."""

    model_config = ConfigDict(from_attributes=True)

    user: UserInfo

class TenantClientSchema(BaseModel):
    """Tenant client schema."""

    id: UUID4
    client_id: str
    client_named_id: str
    client_secret: str

class UserCreate(UserBase):
    """Create user schema"""

    password: str = Field(min_length=8, max_length=100)
    permissions: List[PermissionList] | None = None
    role: UserRoleEnum

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        is_valid, message = validate_password_string(value)
        if not is_valid:
            raise ValueError(message)
        return value

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: UserRoleEnum) -> UserRoleEnum:
        if value == UserRoleEnum.SUPER_ADMIN:
            raise ValueError("The SUPER_ADMIN role is not permitted.")
        return value


class UserFilter(BaseModel):
    """Filter user schema"""

    name: str | None = None
    email: str | None = None
    role: UserRoleEnum | None = None

class UserUpdate(BaseModel):
    """Update user schema"""

    name: str | None = Field(None, min_length=1, max_length=100)
    password: str | None = Field(None, min_length=8, max_length=100)
    role: Optional[UserRoleEnum] = None

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        is_valid, message = validate_password_string(value)
        if not is_valid:
            raise ValueError(message)
        return value

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: Optional[UserRoleEnum]) -> Optional[UserRoleEnum]:
        if value == UserRoleEnum.SUPER_ADMIN:
            raise ValueError("The SUPER_ADMIN role is not permitted.")
        return value


class MyPermissions(SuccessResponse):
    """User permissions schema"""

    model_config = ConfigDict(from_attributes=True)
    permissions: List[Dict[str, Union[str, List[str]]]] = []


class UserListResponse(PaginatedSuccessResponse):
    """User list response to client schema"""

    model_config = ConfigDict(extra="ignore")

    users: list[UserInfo] = []


class UserListFilter(UserFilter):
    """Filter user list schema"""

    status: UserStatusEnum | None = None
