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

"""The main entry point for the application, initializing the FastAPI app and setting up the application's lifespan management, including configuration and secret syncs."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles

from .auth import auth_routes
from .cluster_ops import cluster_routes
from .commons import logging
from .commons.config import app_settings
from .commons.constants import Environment
from .core import common_routes, meta_routes, notify_routes
from .initializers.seeder import seeders
from .model_ops import model_routes
from .user_ops import user_routes
from .workflow_ops import workflow_routes
from .endpoint_ops import endpoint_routes
from .project_ops import project_routes


logger = logging.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage the lifespan of the FastAPI application, including scheduling periodic syncs of configurations and secrets.

    This context manager starts a background task that periodically syncs configurations and secrets from
    their respective stores if they are configured. The sync intervals are randomized between 90% and 100%
    of the maximum sync interval specified in the application settings. The task is canceled upon exiting the
    context.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Yields control back to the context where the lifespan management is performed.
    """

    async def schedule_secrets_and_config_sync() -> None:
        from random import randint

        await asyncio.sleep(3)

        await meta_routes.register_service()
        while True:
            await meta_routes.sync_configurations()
            await meta_routes.sync_secrets()

            await asyncio.sleep(
                randint(
                    int(app_settings.max_sync_interval * 0.9),
                    app_settings.max_sync_interval,
                )
            )

    if app_settings.configstore_name or app_settings.secretstore_name:
        task = asyncio.create_task(schedule_secrets_and_config_sync())
    else:
        task = None

    for seeder_name, seeder in seeders.items():
        try:
            await seeder().seed()
            logger.info(f"Seeded {seeder_name} seeder successfully.")
        except Exception as e:
            logger.error(f"Failed to seed {seeder_name}. Error: {e}")

    yield

    if task is not None:
        try:
            task.cancel()
        except asyncio.CancelledError:
            logger.exception("Failed to cleanup config & store sync.")


app = FastAPI(
    title=app_settings.name,
    description=app_settings.description,
    version=app_settings.version,
    root_path=app_settings.api_root,
    lifespan=lifespan,
    openapi_url=None if app_settings.env == Environment.PRODUCTION else "/openapi.json",
)

# Serve static files
app.mount("/static", StaticFiles(directory=app_settings.static_dir), name="static")

# Set all CORS enabled origins
if app_settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin).strip("/") for origin in app_settings.cors_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

internal_router = APIRouter()
internal_router.include_router(auth_routes.auth_router)
internal_router.include_router(cluster_routes.cluster_router)
internal_router.include_router(common_routes.common_router)
internal_router.include_router(endpoint_routes.endpoint_router)
internal_router.include_router(meta_routes.meta_router)
internal_router.include_router(model_routes.model_router)
internal_router.include_router(notify_routes.notify_router)
internal_router.include_router(user_routes.user_router)
internal_router.include_router(workflow_routes.workflow_router)
internal_router.include_router(project_routes.project_router)

app.include_router(internal_router)


# Override schemas for Swagger documentation
app.openapi_schema = None  # Clear the cached schema


def custom_openapi() -> Any:
    """Customize the OpenAPI schema for Swagger documentation.

    This function modifies the OpenAPI schema to include both API and PubSub models for routes that are marked as PubSub API endpoints.
    This approach allows the API to handle both direct API calls and PubSub events using the same endpoint, while providing clear documentation for API users in the Swagger UI.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    for route in app.routes:
        if hasattr(route, "endpoint") and hasattr(route.endpoint, "is_pubsub_api"):
            request_model = route.endpoint.request_model
            path = route.path
            method = list(route.methods)[0].lower()

            pubsub_model = request_model.create_pubsub_model()
            api_model = request_model.create_api_model()

            openapi_schema["components"]["schemas"][pubsub_model.__name__] = pubsub_model.model_json_schema()
            openapi_schema["components"]["schemas"][api_model.__name__] = api_model.model_json_schema()

            openapi_schema["components"]["schemas"][request_model.__name__] = {
                "oneOf": [
                    {"$ref": f"#/components/schemas/{api_model.__name__}"},
                    {"$ref": f"#/components/schemas/{pubsub_model.__name__}"},
                ]
            }

            openapi_schema["paths"][path][method]["requestBody"]["content"]["application/json"]["schema"] = {
                "$ref": f"#/components/schemas/{api_model.__name__}"
            }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
