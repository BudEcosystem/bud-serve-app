import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from budapp.commons import logging
from uuid import UUID
from cachetools import TTLCache

logger = logging.get_logger(__name__)


class ClusterMetricsFetcher:
    """Fetches cluster metrics from Prometheus."""

    def __init__(self, prometheus_url: str):
        """Initialize with Prometheus server URL."""
        self.prometheus_url = prometheus_url
        self.api_url = f"{prometheus_url}/api/v1"
        self._time_range_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes TTL

    def _get_time_range_params(self, time_range: str) -> dict:
        """Get start and end timestamps based on time range."""
        now = datetime.now(timezone.utc)

        if time_range == "10min":
            start_time = now - timedelta(minutes=10)
            step = "30s"
        elif time_range == "today":
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            step = "5m"
        elif time_range == "7days":
            start_time = now - timedelta(days=7)
            step = "1h"
        elif time_range == "month":
            start_time = now - timedelta(days=30)
            step = "6h"
        else:
            start_time = now - timedelta(minutes=10)
            step = "30s"

        return {"start": start_time.timestamp(), "end": now.timestamp(), "step": step}

    def _get_metric_queries(self, cluster_name: str) -> Dict[str, str]:
        """Get memory-related Prometheus queries."""
        return {
            "memory_total": f'node_memory_MemTotal_bytes{{cluster="{cluster_name}"}} / 1024 / 1024 / 1024',
            "memory_available": f'node_memory_MemAvailable_bytes{{cluster="{cluster_name}"}} / 1024 / 1024 / 1024',
        }

    async def _fetch_metrics(self, session: aiohttp.ClientSession, query: str, time_params: dict) -> List[Dict]:
        """Fetch metrics from Prometheus."""
        try:
            async with session.get(
                f"{self.api_url}/query_range",
                params={
                    "query": query,
                    "start": time_params["start"],
                    "end": time_params["end"],
                    "step": time_params["step"],
                },
                timeout=10,
            ) as response:
                response.raise_for_status()
                result = await response.json()
                return result["data"]["result"]
        except Exception as e:
            logger.error(f"Failed to fetch metrics: {e}")
            return []

    def _process_metrics(self, results: Dict[str, List]) -> Dict[str, Any]:
        """Process memory metrics results."""
        metrics = {
            "nodes": {},
            "cluster_summary": {"memory": {"total_gib": 0, "used_gib": 0, "available_gib": 0, "usage_percent": 0}},
        }

        # Process memory metrics for each node
        if "memory_total" in results and "memory_available" in results:
            for node_data in results["memory_total"]:
                instance = node_data["metric"]["instance"]
                total_values = node_data["values"]

                # Find corresponding available memory
                available_values = []
                for avail_data in results["memory_available"]:
                    if avail_data["metric"]["instance"] == instance:
                        available_values = avail_data["values"]
                        break

                # Process current values
                total = float(total_values[-1][1])
                available = float(available_values[-1][1]) if available_values else 0
                used = total - available

                # Update node metrics
                if instance not in metrics["nodes"]:
                    metrics["nodes"][instance] = {"memory": {}}

                metrics["nodes"][instance]["memory"].update(
                    {
                        "total_gib": total,
                        "used_gib": used,
                        "available_gib": available,
                        "usage_percent": (used / total * 100) if total > 0 else 0,
                    }
                )

                # Update cluster summary
                metrics["cluster_summary"]["memory"]["total_gib"] += total
                metrics["cluster_summary"]["memory"]["used_gib"] += used
                metrics["cluster_summary"]["memory"]["available_gib"] += available

            # Calculate cluster memory usage percentage
            if metrics["cluster_summary"]["memory"]["total_gib"] > 0:
                metrics["cluster_summary"]["memory"]["usage_percent"] = (
                    metrics["cluster_summary"]["memory"]["used_gib"]
                    / metrics["cluster_summary"]["memory"]["total_gib"]
                    * 100
                )

            # Calculate historical usage data
            if results["memory_total"] and results["memory_available"]:
                total_series = results["memory_total"][0]["values"]
                available_series = results["memory_available"][0]["values"]
                
                historical_usage = []
                for i in range(len(total_series)):
                    timestamp = total_series[i][0]
                    total_val = float(total_series[i][1])
                    available_val = float(available_series[i][1])
                    usage_percent = ((total_val - available_val) / total_val * 100) if total_val > 0 else 0
                    historical_usage.append({
                        "timestamp": timestamp,
                        "value": round(usage_percent, 2)
                    })
                
                metrics["historical_data"] = {"memory_usage_percent": historical_usage}

        return metrics

    async def get_cluster_metrics(self, cluster_id: UUID, time_range: str = "today") -> Optional[Dict[str, Any]]:
        """Fetch cluster memory metrics.

        Args:
            cluster_id: The cluster ID to fetch metrics for
            time_range: One of '10min', 'today', '7days', 'month'

        Returns:
            Dict with node and summary metrics for memory
        """
        if not cluster_id:
            logger.error("Cluster ID is required to fetch cluster metrics")
            return None

        cluster_name = str(cluster_id)
        time_params = self._get_time_range_params(time_range)
        queries = self._get_metric_queries(cluster_name)

        try:
            # Fetch metrics concurrently
            async with aiohttp.ClientSession() as session:
                tasks = []
                for metric_name, query in queries.items():
                    task = self._fetch_metrics(session, query, time_params)
                    tasks.append((metric_name, task))

                # Gather results
                results = {}
                for metric_name, task in tasks:
                    try:
                        results[metric_name] = await task
                    except Exception as e:
                        logger.error(f"Failed to fetch {metric_name}: {e}")
                        results[metric_name] = []

            # Process metrics
            processed_metrics = self._process_metrics(results)

            # Add metadata
            processed_metrics.update(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "time_range": time_range,
                    "cluster_id": str(cluster_id),
                }
            )

            logger.info(f"Successfully fetched memory metrics for cluster {cluster_id}")
            return processed_metrics

        except Exception as e:
            logger.error(f"Failed to fetch cluster metrics: {e}")
            return None
