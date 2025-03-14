"""Services for handling credential and cloud provider operations."""

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin
from budapp.credential_ops.crud import CloudProviderDataManager, CloudProviderCredentialDataManager
from budapp.credential_ops.models import CloudCredentials, CloudProviders
from budapp.credential_ops.schemas import CloudProvidersCreateRequest
import json
import uuid
from fastapi import HTTPException, status
from typing import Dict, Any, Union
from uuid import UUID

logger = logging.get_logger(__name__)


class ClusterProviderService(SessionMixin):
    """ClusterProviderService is a service class that provides cluster-related operations."""

    async def create_provider_credential(self, req: CloudProvidersCreateRequest, current_user_id: UUID) -> None:
        """
        Create a new credential for a provider.

        Args:
            req: CloudProvidersCreateRequest containing provider_id and credential_values

        Raises:
            ValueError: If provider is not found or required fields are missing
            HTTPException: If there are validation errors
        """
        try:
            # Convert provider_id string to UUID if needed
            provider_id = req.provider_id
            if not isinstance(provider_id, uuid.UUID):
                try:
                    provider_id = uuid.UUID(provider_id)
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid provider_id format: {provider_id}"
                    )

            # Get the provider from the database
            provider = await CloudProviderDataManager(self.session).retrieve_by_fields(
                CloudProviders, {"id": provider_id}
            )

            # Validate the provider
            if not provider:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Provider with id {provider_id} not found"
                )

            # Handle schema_definition which might be a dict or a JSON string
            schema = self._get_schema_definition(provider.schema_definition)

            # Get the required fields from the schema
            required_fields = schema.get("required", [])

            # Validate the required fields
            for field in required_fields:
                if field not in req.credential_values:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Required field '{field}' is missing in the credential values"
                    )



            # Save the credential values
            cloud_credential = CloudCredentials(
                user_id=current_user_id,
                provider_id=provider_id,
                credential=req.credential_values,
                credential_name=req.credential_name
            )
            await CloudProviderDataManager(self.session).insert_one(cloud_credential)

            logger.debug(f"Created credential for provider {cloud_credential.id}")
        except HTTPException:
            # Re-raise HTTP exceptions without additional logging
            raise
        except Exception as e:
            logger.error(f"Failed to create credential for provider {req.provider_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create credential: {str(e)}"
            )

    def _get_schema_definition(self, schema_definition: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Parse the schema_definition which could be a dict or a JSON string.

        Args:
            schema_definition: The schema definition as either a dict or JSON string

        Returns:
            Dict containing the parsed schema

        Raises:
            ValueError: If the schema_definition is invalid
        """
        if isinstance(schema_definition, dict):
            return schema_definition
        elif isinstance(schema_definition, str):
            try:
                return json.loads(schema_definition)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid schema_definition JSON: {e}")
                raise ValueError(f"Invalid schema_definition: {e}")
        else:
            logger.error(f"Unexpected schema_definition type: {type(schema_definition)}")
            return {}  # Return empty dict as fallback
