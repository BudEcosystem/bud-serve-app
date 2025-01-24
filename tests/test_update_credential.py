from typing import Dict, Any
from datetime import datetime
import os
import aiohttp

from budapp.credential_ops.schemas import CredentialUpdatePayload, CredentialUpdateRequest

async def add_request_metrics_endpoint(payload: Dict[str, Any]):
    endpoint = f"http://localhost:{os.getenv('APP_PORT')}/credentials/update"
    async with aiohttp.ClientSession() as session, session.post(endpoint, json=payload) as response:
        return await response.json()

async def test_update_credential():
    payload = CredentialUpdatePayload(
        hashed_key="cb742dc90b3c735da84104d09715fde454e12bff5f6c7336c1e655628fe9d957",
        last_used_at=datetime.now()
    )
    data = CredentialUpdateRequest(payload=payload).model_dump(mode="json")
    await add_request_metrics_endpoint(data)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_update_credential())
