"""Grafana API client for interacting with Grafana's REST API."""

from typing import Any, Dict, List, Optional

import httpx

from .config import app_settings


class GrafanaClient:
    """Client for interacting with Grafana's REST API."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """Initialize the Grafana client.

        Args:
            api_key: Grafana API key
            base_url: Base URL for Grafana instance. If not provided, uses the one from app settings.
        """
        self.base_url = base_url or app_settings.grafana_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def _request(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request to the Grafana API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request data for POST/PUT requests

        Returns:
            Response data from the API

        Raises:
            httpx.HTTPError: If the request fails
        """
        url = f"{self.base_url}/api{endpoint}"
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                headers=self.headers,
                json=data,
            )
            response.raise_for_status()
            return response.json()

    async def create_dashboard(self, dashboard: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new dashboard.

        Args:
            dashboard: Dashboard configuration

        Returns:
            Created dashboard data
        """
        data = {
            "dashboard": dashboard,
            "overwrite": False,
        }
        return await self._request("POST", "/dashboards/db", data)

    async def get_dashboard(self, uid: str) -> Dict[str, Any]:
        """Get a dashboard by UID.

        Args:
            uid: Dashboard UID

        Returns:
            Dashboard data
        """
        return await self._request("GET", f"/dashboards/uid/{uid}")

    async def delete_dashboard(self, uid: str) -> Dict[str, Any]:
        """Delete a dashboard by UID.

        Args:
            uid: Dashboard UID

        Returns:
            Deletion confirmation
        """
        return await self._request("DELETE", f"/dashboards/uid/{uid}")

    async def create_datasource(self, datasource: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new data source.

        Args:
            datasource: Data source configuration

        Returns:
            Created data source data
        """
        return await self._request("POST", "/datasources", datasource)

    async def get_datasources(self) -> List[Dict[str, Any]]:
        """Get all data sources.

        Returns:
            List of data sources
        """
        return await self._request("GET", "/datasources")

    async def get_datasource(self, id: int) -> Dict[str, Any]:
        """Get a data source by ID.

        Args:
            id: Data source ID

        Returns:
            Data source data
        """
        return await self._request("GET", f"/datasources/{id}")

    async def delete_datasource(self, id: int) -> Dict[str, Any]:
        """Delete a data source by ID.

        Args:
            id: Data source ID

        Returns:
            Deletion confirmation
        """
        return await self._request("DELETE", f"/datasources/{id}")

    async def search_dashboards(
        self, query: str, tag: Optional[str] = None, starred: bool = False
    ) -> List[Dict[str, Any]]:
        """Search for dashboards.

        Args:
            query: Search query
            tag: Filter by tag
            starred: Only return starred dashboards

        Returns:
            List of matching dashboards
        """
        params = {
            "query": query,
            "tag": tag,
            "starred": "true" if starred else "false",
        }
        return await self._request("GET", "/search", params)

    async def get_dashboard_permissions(self, uid: str) -> List[Dict[str, Any]]:
        """Get dashboard permissions.

        Args:
            uid: Dashboard UID

        Returns:
            List of permissions
        """
        return await self._request("GET", f"/dashboards/uid/{uid}/permissions")

    async def update_dashboard_permissions(
        self, uid: str, permissions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update dashboard permissions.

        Args:
            uid: Dashboard UID
            permissions: List of permissions to set

        Returns:
            Updated permissions
        """
        return await self._request(
            "POST", f"/dashboards/uid/{uid}/permissions", {"items": permissions}
        )
