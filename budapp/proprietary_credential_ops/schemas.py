from typing import Optional
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, ValidationInfo, field_validator

from ..commons.constants import PROPRIETARY_CREDENTIAL_DATA, CredentialTypeEnum


class ProprietaryCredentialRequest(BaseModel):
    """Proprietary credential request schema."""

    name: str
    type: CredentialTypeEnum
    other_provider_creds: Optional[dict] = {}

    @field_validator("other_provider_creds")
    @classmethod
    def validate_other_provider_creds(cls, v, info: ValidationInfo):
        """Validate the proprietary credentials."""
        # Validate the proprietary credentials using the common validator
        v = common_validator_for_provider_creds(v, info)

        return v


class ProprietaryCredentialResponse(BaseModel):
    """Proprietary Credential response schema."""

    name: str
    type: CredentialTypeEnum
    other_provider_creds: dict | None
    id: UUID


class ProprietaryCredentialFilter(BaseModel):
    """Filter proprietary credential schema."""

    name: str | None = None
    type: CredentialTypeEnum | None = None
    id: UUID | None = None


class ProprietaryCredentialUpdate(BaseModel):
    """Proprietary Credential update schema."""

    name: str | None = None
    type: CredentialTypeEnum
    other_provider_creds: dict | None = None
    endpoint_id: UUID | None = None

    @field_validator("other_provider_creds")
    @classmethod
    def validate_other_provider_creds(cls, v, info: ValidationInfo):
        """Validate the proprietary credentials."""
        # Validate the proprietary credentials using the common validator
        v = common_validator_for_provider_creds(v, info)

        return v


def common_validator_for_provider_creds(v, info: ValidationInfo):
    """Validate provider credentials using common validation rules."""
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
