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
from typing import List

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
                        status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid provider_id format: {provider_id}"
                    )

            # Get the provider from the database
            provider = await CloudProviderDataManager(self.session).retrieve_by_fields(
                CloudProviders, {"id": provider_id}
            )

            # Validate the provider
            if not provider:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=f"Provider with id {provider_id} not found"
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
                        detail=f"Required field '{field}' is missing in the credential values",
                    )

            # Save the credential values
            cloud_credential = CloudCredentials(
                user_id=current_user_id,
                provider_id=provider_id,
                credential=req.credential_values,
                credential_name=req.credential_name,
            )
            await CloudProviderDataManager(self.session).insert_one(cloud_credential)

            logger.debug(f"Created credential for provider {cloud_credential.id}")
        except HTTPException:
            # Re-raise HTTP exceptions without additional logging
            raise
        except Exception as e:
            logger.error(f"Failed to create credential for provider {req.provider_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create credential: {str(e)}"
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

    async def get_provider_regions(self, unique_id: str) -> List[Dict[str, Any]]:
        """
        Get the regions supported by a specific cloud provider.

        Args:
            provider: The cloud provider entity

        Returns:
            List of regions as dictionaries with at least 'id' and 'name' keys
        """

        provider_regions = {
            "aws": [
                {"id": "us-east-1", "name": "US East (N. Virginia)"},
                {"id": "us-east-2", "name": "US East (Ohio)"},
                {"id": "us-west-1", "name": "US West (N. California)"},
                {"id": "us-west-2", "name": "US West (Oregon)"},
                {"id": "ca-central-1", "name": "Canada (Central)"},
                {"id": "ca-west-1", "name": "Canada West (Calgary)"},
                {"id": "sa-east-1", "name": "South America (São Paulo)"},
                {"id": "eu-west-1", "name": "Europe (Ireland)"},
                {"id": "eu-west-2", "name": "Europe (London)"},
                {"id": "eu-west-3", "name": "Europe (Paris)"},
                {"id": "eu-central-1", "name": "Europe (Frankfurt)"},
                {"id": "eu-north-1", "name": "Europe (Stockholm)"},
                {"id": "eu-south-1", "name": "Europe (Milan)"},
                {"id": "eu-central-2", "name": "Europe (Zurich)"},
                {"id": "eu-south-2", "name": "Europe (Spain)"},
                {"id": "ap-northeast-1", "name": "Asia Pacific (Tokyo)"},
                {"id": "ap-northeast-2", "name": "Asia Pacific (Seoul)"},
                {"id": "ap-northeast-3", "name": "Asia Pacific (Osaka)"},
                {"id": "ap-southeast-1", "name": "Asia Pacific (Singapore)"},
                {"id": "ap-southeast-2", "name": "Asia Pacific (Sydney)"},
                {"id": "ap-east-1", "name": "Asia Pacific (Hong Kong)"},
                {"id": "ap-south-1", "name": "Asia Pacific (Mumbai)"},
                {"id": "ap-southeast-3", "name": "Asia Pacific (Jakarta)"},
                {"id": "ap-southeast-4", "name": "Asia Pacific (Melbourne)"},
                {"id": "ap-south-2", "name": "Asia Pacific (Hyderabad)"},
                {"id": "ap-southeast-5", "name": "Asia Pacific (Malaysia)"},
                {"id": "me-south-1", "name": "Middle East (Bahrain)"},
                {"id": "me-central-1", "name": "Middle East (UAE)"},
                {"id": "il-central-1", "name": "Israel (Tel Aviv)"},
                {"id": "af-south-1", "name": "Africa (Cape Town)"},
                {"id": "cn-north-1", "name": "China (Beijing)"},
                {"id": "cn-northwest-1", "name": "China (Ningxia)"},
                {"id": "us-gov-west-1", "name": "AWS GovCloud (US-West)"},
                {"id": "us-gov-east-1", "name": "AWS GovCloud (US-East)"},
            ],
            "azure": [
                {"id": "eastus", "name": "East US"},
                {"id": "eastus2", "name": "East US 2"},
                {"id": "southcentralus", "name": "South Central US"},
                {"id": "westus", "name": "West US"},
                {"id": "westus2", "name": "West US 2"},
                {"id": "westus3", "name": "West US 3"},
                {"id": "centralus", "name": "Central US"},
                {"id": "canadacentral", "name": "Canada Central"},
                {"id": "canadaeast", "name": "Canada East"},
                {"id": "brazilsouth", "name": "Brazil South"},
                {"id": "uksouth", "name": "UK South"},
                {"id": "ukwest", "name": "UK West"},
                {"id": "francecentral", "name": "France Central"},
                {"id": "francesouth", "name": "France South"},
                {"id": "germanywestcentral", "name": "Germany West Central"},
                {"id": "germanynorth", "name": "Germany North"},
                {"id": "switzerlandnorth", "name": "Switzerland North"},
                {"id": "switzerlandwest", "name": "Switzerland West"},
                {"id": "norwayeast", "name": "Norway East"},
                {"id": "norwaywest", "name": "Norway West"},
                {"id": "australiaeast", "name": "Australia East"},
                {"id": "australiasoutheast", "name": "Australia Southeast"},
                {"id": "australiacentral", "name": "Australia Central"},
                {"id": "australiacentral2", "name": "Australia Central 2"},
                {"id": "japaneast", "name": "Japan East"},
                {"id": "japanwest", "name": "Japan West"},
                {"id": "koreacentral", "name": "Korea Central"},
                {"id": "koreasouth", "name": "Korea South"},
                {"id": "southeastasia", "name": "Southeast Asia"},
                {"id": "eastasia", "name": "East Asia"},
                {"id": "centralindia", "name": "Central India"},
                {"id": "southindia", "name": "South India"},
                {"id": "westindia", "name": "West India"},
                {"id": "uaenorth", "name": "UAE North"},
                {"id": "uaecentral", "name": "UAE Central"},
                {"id": "southafricanorth", "name": "South Africa North"},
                {"id": "southafricawest", "name": "South Africa West"},
                {"id": "qatarcentral", "name": "Qatar Central"},
                {"id": "israelcentral", "name": "Israel Central"}
              ]
        }

        # Match based on the unique_id
        if unique_id in provider_regions:
            return provider_regions[unique_id]

        return []
