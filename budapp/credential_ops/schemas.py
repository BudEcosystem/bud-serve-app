from datetime import datetime
from enum import IntEnum
from typing import Annotated, Dict, List, Optional, Union
from uuid import UUID

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    confloat,
    field_validator,
    model_validator,
)

from budapp.commons import logging

from ..commons.config import app_settings
from ..commons.constants import CredentialTypeEnum
from ..commons.schemas import CloudEventBase, PaginatedSuccessResponse
from ..initializers.provider_seeder import ProviderSeeder


logger = logging.get_logger(__name__)

PROPRIETARY_CREDENTIAL_DATA = ProviderSeeder._get_providers_data()

model_budget_type = Annotated[
    Dict[UUID, confloat(gt=0)], "A dictionary with UUID keys and float values greater than 0"
]


class CredentialUpdatePayload(BaseModel):
    hashed_key: str
    last_used_at: datetime


class CredentialUpdateRequest(CloudEventBase):
    """Request to update the credential last used at time."""

    model_config = ConfigDict(extra="allow")

    payload: CredentialUpdatePayload

    @model_validator(mode="before")
    def log_credential_update(cls, data):
        """Log the credential update hits for debugging purposes."""
        # TODO: remove this function after Debugging
        logger.info("================================================")
        logger.info("Received hit in credentials/update:")
        logger.info(f"{data}")
        logger.info("================================================")
        return data


class CredentialBase(BaseModel):
    """Credential base schema"""

    key: str


class CredentialCreate(CredentialBase):
    """Create credential schema"""

    user_id: UUID


class CredentialFilter(BaseModel):
    """Credential filter schema"""

    name: Optional[str] = None
    project_id: Optional[UUID] = None


class BudCredentialCreate(CredentialCreate):
    """Create credential schema"""

    name: str
    project_id: UUID
    expiry: datetime | None
    max_budget: float | None
    model_budgets: Optional[model_budget_type] = None

    model_config = ConfigDict(protected_namespaces=())


class ExpiryEnum(IntEnum):
    """Expiry enum"""

    THIRTY_DAYS = 30
    SIXTY_DAYS = 60


class CredentialRequest(BaseModel):
    """Credential request schema"""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    project_id: UUID
    expiry: Optional[ExpiryEnum] = None
    max_budget: Optional[float] = Field(
        default=None, gt=0, description="Max budget must be greater than 0 if specified"
    )
    model_budgets: Optional[model_budget_type] = None

    @model_validator(mode="after")
    def validate_key(self) -> "CredentialRequest":
        if self.max_budget and self.model_budgets and (sum(self.model_budgets.values()) > self.max_budget):
            raise ValueError("Sum of model budgets should not exceed max budget")
        return self


class Credential(BudCredentialCreate):
    """Credential schema"""

    id: UUID
    created_at: datetime
    modified_at: datetime


class CredentialResponse(BaseModel):
    """Credential response schema"""

    name: str
    key: str
    project_id: UUID
    expiry: datetime | None
    max_budget: float | None
    model_budgets: Optional[model_budget_type] = None
    id: UUID

    model_config = ConfigDict(protected_namespaces=())


class CredentialProject(BaseModel):
    """Credential project schema"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str


class CredentialDetails(BaseModel):
    """BudServe credential details"""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    project: CredentialProject
    key: str
    expiry: datetime | None
    max_budget: float | None
    model_budgets: Optional[model_budget_type] = None
    id: UUID
    last_used_at: datetime | None


class CredentialUpdate(BaseModel):
    """Credential Update schema"""

    model_config = ConfigDict(protected_namespaces=())

    name: str | None = None
    expiry: Optional[ExpiryEnum] = None
    max_budget: float | None = None
    model_budgets: Optional[model_budget_type] = None

    @model_validator(mode="after")
    def validate_key(self) -> "CredentialUpdate":
        if self.max_budget and self.model_budgets and (sum(self.model_budgets.values()) > self.max_budget):
            raise ValueError("Sum of model budgets should not exceed max budget")
        return self


class ProprietaryCredentialRequest(BaseModel):
    """Proprietary credential request schema"""

    name: str
    type: CredentialTypeEnum
    provider_id: Optional[UUID] = None
    other_provider_creds: Optional[dict] = {}

    @field_validator("other_provider_creds")
    @classmethod
    def validate_other_provider_creds(cls, v, info: ValidationInfo):
        # Validate the proprietary credentials using the common validator
        v = common_validator_for_provider_creds(v, info)

        return v


class ProprietaryCredentialResponse(BaseModel):
    """Proprietary Credential response schema"""

    name: str
    type: CredentialTypeEnum
    other_provider_creds: dict | None
    id: UUID


class ProprietaryCredentialFilter(BaseModel):
    """Filter proprietary credential schema"""

    name: str | None = None
    type: CredentialTypeEnum | None = None
    id: UUID | None = None


class ProprietaryCredentialUpdate(BaseModel):
    """Proprietary Credential update schema"""

    name: str | None = None
    type: CredentialTypeEnum
    other_provider_creds: dict | None = None
    endpoint_id: UUID | None = None

    @field_validator("other_provider_creds")
    @classmethod
    def validate_other_provider_creds(cls, v, info: ValidationInfo):
        # Validate the proprietary credentials using the common validator
        v = common_validator_for_provider_creds(v, info)

        return v


class ProprietaryCredentialResponseList(ProprietaryCredentialResponse):
    """Proprietary Credential response list schema"""

    created_at: datetime
    num_of_endpoints: int
    provider_icon: str


class ProprietaryCredentialDetailedView(ProprietaryCredentialResponseList):
    """Proprietary Credential detailed view schema"""

    endpoints: list


class PaginatedCredentialResponse(PaginatedSuccessResponse):
    """Paginated Credential response schema"""

    credentials: List[Union[ProprietaryCredentialResponseList, CredentialDetails]]


def common_validator_for_provider_creds(v, info: ValidationInfo):
    """Common validator for provider credentials"""

    credential_type = info.data.get("type")
    if not credential_type:
        raise ValueError("Credential type must be specified")

    credential_data = PROPRIETARY_CREDENTIAL_DATA[credential_type.value]
    expected_fields = credential_data.get("credentials", [])

    # Check if invalid fields are provided
    expected_fields_names = [field["field"] for field in expected_fields]
    for key in v:
        if key not in expected_fields_names:
            raise ValueError(f"Unexpected field: {key}")

    for field in expected_fields:
        field_name = field["field"]

        # Check if the field is required
        if field["required"] and field_name not in v:
            raise ValueError(f"Missing required field: {field_name}")

        if field_name in v:
            value = v[field_name]
            field_type = field["type"]

            # Check if the field is a dependency
            field_dependencies = field.get("dependencies", [])
            for dependency in field_dependencies:
                if dependency not in v:
                    raise ValueError(f"Missing dependency: {dependency}")

            # Type validation
            if field_type in ["text", "password", "textarea"]:
                if not isinstance(value, str):
                    raise ValueError(f"Field {field_name} must be a string")
            elif field_type == "url":
                AnyHttpUrl(value)
            # TODO: Add more type validations here

    return v


class CacheConfig(BaseModel):
    embedding_model: Optional[str] = app_settings.cache_embedding_model
    eviction_policy: Optional[str] = app_settings.cache_eviction_policy
    max_size: Optional[int] = app_settings.cache_max_size
    ttl: Optional[int] = app_settings.cache_ttl
    score_threshold: Optional[float] = app_settings.cache_score_threshold


class ModelConfig(BaseModel):
    """Model config schema"""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    litellm_params: dict
    model_info: dict


class RouterConfig(BaseModel):
    """Router config schema"""

    model_config = ConfigDict(protected_namespaces=())

    project_id: UUID
    project_name: str
    endpoint_name: str
    cache_configuration: CacheConfig | None
    model_configuration: ModelConfig | None
