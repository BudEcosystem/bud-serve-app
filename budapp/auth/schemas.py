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


"""Contains Pydantic schemas used for data validation and serialization within the auth services."""

from pydantic import UUID4, BaseModel, ConfigDict, EmailStr, Field

from budapp.commons.constants import TokenTypeEnum
from budapp.commons.schemas import SuccessResponse


class AuthToken(BaseModel):
    """Token schema."""

    access_token: str
    refresh_token: str
    token_type: str  # Bearer


class AccessTokenData(BaseModel):
    """Access token data schema."""

    sub: str
    type: str = TokenTypeEnum.ACCESS.value


class RefreshTokenData(BaseModel):
    """Refresh token data schema."""

    sub: str
    type: str = TokenTypeEnum.REFRESH.value
    secret_key: str


class TokenCreate(BaseModel):
    """Token create schema."""

    auth_id: UUID4
    secret_key: str | None = None
    token_hash: str
    type: TokenTypeEnum


class UserLogin(BaseModel):
    """Login user schema."""

    email: EmailStr = Field(min_length=1, max_length=100)
    password: str = Field(min_length=8, max_length=100)


class UserLoginData(BaseModel):
    """User login data schema."""

    token: AuthToken
    first_login: bool
    is_reset_password: bool


class UserLoginResponse(SuccessResponse):
    """User login response schema."""

    model_config = ConfigDict(extra="ignore")
    token: AuthToken
    first_login: bool
    is_reset_password: bool
