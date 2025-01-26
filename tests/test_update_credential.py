from typing import Dict, Any
from datetime import datetime
from unittest.mock import AsyncMock, patch
import os
import aiohttp
import pytest

from budapp.credential_ops.schemas import CredentialUpdatePayload, CredentialUpdateRequest
from budapp.shared.dapr_service import DaprService
from budapp.shared.redis_service import RedisService
from budapp.commons.db_utils import Session


async def add_request_metrics_endpoint(payload: Dict[str, Any]):
    endpoint = f"http://localhost:{os.getenv('APP_PORT')}/credentials/update"
    async with aiohttp.ClientSession() as session, session.post(endpoint, json=payload) as response:
        return await response.json()


@pytest.mark.asyncio
async def test_update_credential():
    mock_dapr_service = AsyncMock(spec=DaprService)
    mock_redis_service = AsyncMock(spec=RedisService)
    mock_session = AsyncMock(spec=Session)

    with patch("budapp.credential_ops.services.DaprService", return_value=mock_dapr_service), \
            patch("budapp.credential_ops.services.RedisService", return_value=mock_redis_service), \
            patch("budapp.credential_ops.services.Session", return_value=mock_session):
        payload = CredentialUpdatePayload(
            hashed_key="cb742dc90b3c735da84104d09715fde454e12bff5f6c7336c1e655628fe9d957",
            last_used_at=datetime.now()
        )
        data = CredentialUpdateRequest(payload=payload).model_dump(mode="json")
        await add_request_metrics_endpoint(data)

        mock_dapr_service.publish_to_topic.assert_called_once()
        mock_redis_service.set.assert_called_once()
        mock_session.commit.assert_called_once()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_update_credential())
