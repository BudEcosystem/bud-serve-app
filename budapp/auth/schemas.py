from pydantic import UUID4, BaseModel

from budapp.commons.constants import TokenTypeEnum


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
