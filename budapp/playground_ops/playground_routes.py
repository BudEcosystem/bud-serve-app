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

"""The playground ops package, containing essential business logic, services, and routing configurations for the playground ops."""

from typing import List, Union

from fastapi import APIRouter, Depends, Header, Query, status
from fastapi.security.http import HTTPAuthorizationCredentials, get_authorization_scheme_param
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from ..commons import logging
from ..commons.constants import ModalityEnum
from ..commons.dependencies import (
    get_current_active_user,
    get_current_user,
    get_session,
    parse_ordering_fields,
)
from ..commons.exceptions import ClientException
from ..commons.schemas import ErrorResponse
from .schemas import PlaygroundDeploymentFilter, PlaygroundDeploymentListResponse
from .services import PlaygroundService


logger = logging.get_logger(__name__)

playground_router = APIRouter(prefix="/playground", tags=["playground"])


@playground_router.get(
    "/deployments",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
        status.HTTP_200_OK: {
            "model": PlaygroundDeploymentListResponse,
            "description": "Successfully list all playground deployments",
        },
    },
    description="List all playground deployments",
)
async def list_playground_deployments(
    session: Annotated[Session, Depends(get_session)],
    filters: Annotated[PlaygroundDeploymentFilter, Depends()],
    modality: List[ModalityEnum] = Query(default=[]),
    tags: List[str] = Query(default=[]),
    tasks: List[str] = Query(default=[]),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    authorization: Annotated[
        str | None, Header()
    ] = None,  # NOTE: Can't use in Openapi docs https://github.com/fastapi/fastapi/issues/612#issuecomment-547886504
    api_key: Annotated[str | None, Header()] = None,
    order_by: Annotated[List[str] | None, Depends(parse_ordering_fields)] = None,
    search: bool = False,
) -> Union[PlaygroundDeploymentListResponse, ErrorResponse]:
    """List all playground deployments."""
    current_user_id = None
    if authorization:
        scheme, credentials = get_authorization_scheme_param(authorization)
        token = HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)
        db_user = await get_current_user(token, session)
        current_user = await get_current_active_user(db_user)
        current_user_id = current_user.id
    elif not authorization and not api_key:
        raise ClientException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            message="Authorization header or API key is required",
        )

    # Calculate offset
    offset = (page - 1) * limit

    # Convert PlaygroundDeploymentFilter to dictionary
    filters_dict = filters.model_dump(exclude_none=True)

    # Update filters_dict only for non-empty lists
    filter_updates = {"tags": tags, "tasks": tasks, "modality": modality}
    filters_dict.update({k: v for k, v in filter_updates.items() if v})

    try:
        db_endpoints, count = await PlaygroundService(session).get_all_playground_deployments(
            current_user_id,
            api_key,
            offset,
            limit,
            filters_dict,
            order_by,
            search,
        )
        return PlaygroundDeploymentListResponse(
            endpoints=db_endpoints,
            total_record=count,
            page=page,
            limit=limit,
            object="playground.deployments.list",
            code=status.HTTP_200_OK,
        ).to_http_response()
    except ClientException as e:
        logger.exception(f"Failed to get all playground deployments: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get all playground deployments: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get all playground deployments"
        ).to_http_response()
