from typing import Dict, List, Optional
from uuid import UUID

from fastapi import status
from fastapi.exceptions import HTTPException

from ..commons.db_utils import SessionMixin
from ..commons.logging import get_logger
from .crud import ProprietaryCredentialDataManager
from .models import ProprietaryCredential as ProprietaryCredentialModel
from .schemas import ProprietaryCredentialRequest, ProprietaryCredentialResponse, ProprietaryCredentialUpdate


logger = get_logger(__name__)

class ProprietaryCredentialService(SessionMixin):
    async def add_credential(self, current_user_id: UUID, credential: ProprietaryCredentialRequest) -> ProprietaryCredentialResponse:
        """Add a new proprietary credential for the given user.

        Args:
            current_user_id: UUID of the user adding the credential
            credential: The credential details to be added

        Returns:
            ProprietaryCredentialResponse: The newly created credential

        Raises:
            HTTPException: If credential with same name and type already exists
        """
        # Check duplicate credential exists with same name and type for user_id
        db_credential = await ProprietaryCredentialDataManager(self.session).retrieve_credential_by_fields(
            {"name": credential.name, "type": credential.type.value, "user_id": current_user_id}, missing_ok=True
        )

        # Raise error if credential already exists with same name and type
        if db_credential:
            error_msg = f"{credential.type.value} credential already exists with the same name, change name or update existing credential"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg,
            )

        # Encrypt credential and add in db
        db_credential = await self.add_encrypted_credential(credential, current_user_id)

        credential_response = ProprietaryCredentialResponse(
            name=db_credential.name,
            type=db_credential.type,
            id=db_credential.id,
            other_provider_creds=db_credential.other_provider_creds,
        )

        return credential_response

    async def add_encrypted_credential(self, credential: ProprietaryCredentialRequest, user_id: UUID) -> ProprietaryCredentialModel:
        """Encrypt and store a proprietary credential in the database.

        Args:
            credential: The credential details to be encrypted and stored
            user_id: UUID of the user owning the credential

        Returns:
            ProprietaryCredentialModel: The stored credential model
        """
        # Encrypt proprietary credentials
        if credential.other_provider_creds:
            for key, value in credential.other_provider_creds.items():
                credential.other_provider_creds[key] = await RSAHandler().encrypt(value)

        # Insert credential in to database
        credential_model = ProprietaryCredentialModel(**credential.model_dump(), user_id=user_id)
        credential_model.type = credential_model.type.value
        db_credential = await ProprietaryCredentialDataManager(self.session).create_credential(credential_model)
        logger.info(f"Proprietary Credential inserted to database: {db_credential.id}")

        return db_credential

    async def get_all_credentials(
        self, offset: int = 0, limit: int = 10, filters: Optional[Dict] = None, order_by: Optional[List[str]] = None, search: bool = False
    ) -> tuple[list[ProprietaryCredentialResponse], int]:
        """Retrieve all proprietary credentials with pagination and filtering options.

        Args:
            offset: Number of records to skip
            limit: Maximum number of records to return
            filters: Dictionary of filter conditions
            order_by: List of fields to sort by
            search: Whether to enable search functionality

        Returns:
            Tuple containing list of credentials and total count
        """
        filters = filters or {}
        order_by = order_by or []

        if filters.get("type"):
            filters["type"] = filters["type"].value

        db_credentials, count = await ProprietaryCredentialDataManager(self.session).get_all_credentials(
            offset, limit, filters, order_by, search
        )
        return await self._parse_credentials(db_credentials), count

    async def _parse_credentials(
        self,
        db_credentials: List[ProprietaryCredentialModel],
    ) -> List[ProprietaryCredentialResponse]:
        """Parse and decrypt database credentials into response format.

        Args:
            db_credentials: List of credential models from database

        Returns:
            List of formatted credential responses
        """
        # Parse credentials to a common format
        result = []

        # Iterate over credentials and append as per type
        for db_credential in db_credentials:
            if db_credential.other_provider_creds:
                for key, value in db_credential.other_provider_creds.items():
                    db_credential.other_provider_creds[key] = await RSAHandler().decrypt(value)
            result.append(
                ProprietaryCredentialResponse(
                    name=db_credential.name,
                    type=db_credential.type,
                    other_provider_creds=db_credential.other_provider_creds,
                    id=db_credential.id,
                )
            )

        return result

    async def update_credential(
        self,
        credential_id: UUID,
        data: ProprietaryCredentialUpdate,
        current_user_id: UUID,
    ) -> ProprietaryCredentialResponse:
        """Update an existing proprietary credential.

        Args:
            credential_id: UUID of the credential to update
            data: Updated credential information
            current_user_id: UUID of the user performing the update

        Returns:
            ProprietaryCredentialResponse: The updated credential

        Raises:
            HTTPException: If user lacks permission or credential validation fails
        """
        # Check if credential exists
        db_credential = await ProprietaryCredentialDataManager(self.session).retrieve_credential_by_fields({"id": credential_id})

        if db_credential.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User does not have permission to update this credential",
            )

        # Check data type
        if data.type != db_credential.type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Credential type cannot be changed from {db_credential.type} to {data.type.value}",
            )

        proprietary_update_data = data.model_dump(exclude_none=True, exclude={"type"})
        # Handle name
        if proprietary_update_data.get("name", None):
            # Check duplicate credential exists with same name and type for user_id
            db_credential_by_name = await ProprietaryCredentialDataManager(self.session).retrieve_credential_by_fields(
                {"name": data.name, "type": db_credential.type, "user_id": current_user_id}, missing_ok=True
            )

            # Raise error if credential already exists with same name and type
            if db_credential_by_name:
                error_msg = f"Update failed : {db_credential.type} credential already exists with the same name"
                logger.error(error_msg)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg,
                )

        if proprietary_update_data.get("other_provider_creds", None) and data.other_provider_creds:
            # if proprietary_update_data has other_provider_creds,
            # then data will also have attribute other_provider_creds
            # the data.other_provider_creds clause is added to resolve mypy error
            for key, value in data.other_provider_creds.items():
                proprietary_update_data["other_provider_creds"][key] = await RSAHandler().encrypt(value)

        # Handle endpoint
        if proprietary_update_data.get("endpoint_id", None):
            credential_endpoints = db_credential.endpoints
            endpoint_id = data.endpoint_id
            del proprietary_update_data["endpoint_id"]

            # check if endpoint exists in credential endpoints
            for endpoint in credential_endpoints:
                if endpoint.id == endpoint_id:
                    break
            else:
                # Check if endpoint exists
                db_endpoint = await EndpointDataManager(
                    self.session
                ).retrieve_endpoint_by_fields({"id": endpoint_id})
                project_id = db_endpoint.project_id
                # Check user has authority to create credential for project
                db_permission = await PermissionDataManager(
                    self.session
                ).retrieve_permission_by_fields(
                    {"user_id": current_user_id}, missing_ok=True
                )
                global_user_scopes = db_permission.scopes_list if db_permission else []
                if PermissionEnum.PROJECT_MANAGE.value not in global_user_scopes:
                    db_project_permission = await ProjectPermissionDataManager(
                        self.session
                    ).retrieve_project_permission_by_fields(
                        {"user_id": current_user_id, "project_id": project_id},
                        missing_ok=True,
                    )
                    project_user_scopes = (
                        db_project_permission.scopes_list
                        if db_project_permission
                        else []
                    )

                    # Check user has access to endpoint
                    if PermissionEnum.ENDPOINT_MANAGE.value not in project_user_scopes:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="User does not have permission to update credential for this endpoint",
                        )
                db_credential.endpoints.append(db_endpoint)

        # Update the credential in the database
        db_credential = await ProprietaryCredentialDataManager(self.session).update_credential_by_fields(
            db_credential, proprietary_update_data
        )

        return db_credential

    async def delete_credential(self, credential_id: UUID, current_user_id: UUID):
        """Delete the proprietary credential from the database."""
        # Retrieve the credential from the database
        db_credential = await ProprietaryCredentialDataManager(self.session).retrieve_credential_by_fields({"id": credential_id})

        if db_credential.user_id != current_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User does not have permission to delete this credential",
            )

        endpoints = db_credential.endpoints
        if endpoints:
            project_names = [endpoint.project.name for endpoint in endpoints]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"""Credential is associated with deployed models in the below projects : 
                {", ".join(project_names)}.
                Please delete the deployed models first or link other credentials to those models for deleting this credential""",
            )

        # Delete the credential from the database
        await ProprietaryCredentialDataManager(self.session).delete_credential(db_credential)

    async def get_credential_details(self, credential_id: UUID) -> ProprietaryCredentialModel:
        """Get details of a proprietary credential."""
        db_credential = await ProprietaryCredentialDataManager(self.session).retrieve_credential_by_fields({"id": credential_id})
        # Decrypt proprietary credentials
        if db_credential.other_provider_creds:
            for key, value in db_credential.other_provider_creds.items():
                db_credential.other_provider_creds[key] = await RSAHandler().decrypt(value)
        return db_credential
