from datetime import datetime

from pydantic import UUID4, BaseModel, ConfigDict, EmailStr, Field

from budapp.commons.constants import UserRoleEnum, UserStatusEnum


class UserBase(BaseModel):
    """Base user schema."""

    name: str = Field(min_length=1, max_length=100)
    email: EmailStr = Field(min_length=1, max_length=100)


class UserResponse(UserBase):
    """User response to client schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID4
    color: str
    role: UserRoleEnum


class User(UserResponse):
    """User schema."""

    model_config = ConfigDict(from_attributes=True)

    auth_id: UUID4
    password: str
    is_active: bool
    status: UserStatusEnum
    created_at: datetime
    modified_at: datetime
