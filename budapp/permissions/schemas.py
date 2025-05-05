import json
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, field_validator


class RemoveUserScopeFromResource(BaseModel):
    """Remove user scope from resource."""

    resource_type: str
    entity_id: Optional[str] = None
    user_auth_id: UUID
    scope: str

class RemoveUserFromResource(BaseModel):
    """Remove user from resource."""

    resource_type: str
    entity_id: Optional[str] = None
    user_auth_id: UUID

class AssignResourceScopeToUser(BaseModel):
    """Assign resource scope to user."""
    resource_type: str
    entity_id: Optional[str] = None
    user_auth_id: UUID
    scopes: List[str] 

class CheckUserResourceScope(BaseModel):
    """Check user resource scope."""

    resource_type: str
    entity_id: Optional[str] = None
    scope: str # Only One Scope

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


class PermissionUpdate(BaseModel):
    scopes: str

    @field_validator("scopes", mode="before")
    @classmethod
    def convert_json_to_string(cls, v):
        if isinstance(v, list):
            return json.dumps(v)
        elif isinstance(v, str):
            return v
        raise ValueError(v)


class PermissionList(BaseModel):
    name: str
    has_permission: bool


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


class ProjectPermissionBase(BaseModel):
    """Base model for project permission"""

    id: UUID
    permissions: List[PermissionList]


class ProjectPermissionList(ProjectPermissionBase):
    """Project permissions list"""

    name: str


class PermissionListResponse(BaseModel):
    """Permission list response to client schema"""

    global_scopes: List[PermissionList]
    project_scopes: List[ProjectPermissionList]


class ProjectPermissionUpdate(BaseModel):
    """Project permission update"""

    user_id: UUID
    project_id: UUID
    permissions: List[PermissionList]
