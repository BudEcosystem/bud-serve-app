import requests
from typing import Dict, Any, Optional

class ClusterMetricsFetcher:
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
        self.api_url = f"{prometheus_url}/api/v1"

    def query(self, query: str) -> list:
        """Execute a Prometheus query"""
        response = requests.get(
            f"{self.api_url}/query",
            params={'query': query},
            verify=True
        )
        response.raise_for_status()
        return response.json()['data']['result']

    def get_cluster_metrics(self) -> Optional[Dict[str, Any]]:
        #TODO : Implement this method
        pass
