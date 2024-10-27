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
