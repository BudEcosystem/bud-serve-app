import requests
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from budapp.commons import logging
from uuid import UUID
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache, cached
from contextlib import asynccontextmanager

logger = logging.get_logger(__name__)


class ClusterMetricsFetcher:
    """Fetches cluster metrics from Prometheus."""

    def __init__(self, prometheus_url: str):
        """Initialize with Prometheus server URL."""
        self.prometheus_url = prometheus_url
        self.api_url = f"{prometheus_url}/api/v1"
        self._time_range_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes TTL

    @asynccontextmanager
    async def _get_executor(self):
        """Get a thread pool executor context manager."""
        executor = ThreadPoolExecutor(max_workers=10)
        try:
            yield executor
        finally:
            executor.shutdown(wait=True)

    @cached(cache=TTLCache(maxsize=100, ttl=300))  # Cache results for 5 minutes
    def _get_time_range_params(self, time_range: str) -> dict:
        """Get start and end timestamps based on time range.

        Args:
            time_range: One of 'today', '7days', 'month'

        Returns:
            Dict with start and end timestamps and step interval
        """
        now = datetime.now(timezone.utc)

        if time_range == "today":
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            step = "5m"  # 5-minute intervals for today
        elif time_range == "7days":
            start_time = now - timedelta(days=7)
            step = "1h"  # 1-hour intervals for 7 days
        elif time_range == "month":
            start_time = now - timedelta(days=30)
            step = "6h"  # 6-hour intervals for monthly view
        else:
            # Default to today if invalid range
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            step = "5m"

        return {"start": start_time.timestamp(), "end": now.timestamp(), "step": step}

    async def _async_query_range(self, session: aiohttp.ClientSession, query: str, time_params: dict) -> dict:
        """Execute a single Prometheus range query asynchronously."""
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
            logger.error(f"Failed to execute Prometheus range query: {e}")
            return []

    async def _fetch_all_metrics(self, queries: Dict[str, str], time_range: str) -> Dict[str, List]:
        """Fetch all metrics concurrently using asyncio."""
        time_params = self._get_time_range_params(time_range)
        async with aiohttp.ClientSession() as session:
            tasks = []
            for key, query in queries.items():
                task = asyncio.create_task(self._async_query_range(session, query, time_params))
                tasks.append((key, task))

            results = {}
            for key, task in tasks:
                try:
                    results[key] = await task
                except Exception as e:
                    logger.error(f"Failed to fetch metrics for {key}: {e}")
                    results[key] = []

            return results

    def _get_metric_queries(self, cluster_name: str, rate_interval: str, metric_type: str = "all") -> Dict[str, str]:
        """Get Prometheus queries based on metric type.

        Args:
            cluster_name: Name/ID of the cluster
            rate_interval: Rate interval for rate queries
            metric_type: Type of metrics to fetch ('all', 'memory', 'cpu', 'disk', 'gpu', 'hpu', 'network')

        Returns:
            Dict of query names and their Prometheus queries
        """
        queries = {}

        if metric_type in ["all", "memory"]:
            queries.update(
                {
                    "memory_total": f'node_memory_MemTotal_bytes{{cluster="{cluster_name}"}} / 1024 / 1024 / 1024',
                    "memory_available": f'node_memory_MemAvailable_bytes{{cluster="{cluster_name}"}} / 1024 / 1024 / 1024',
                }
            )

        if metric_type in ["all", "cpu"]:
            queries.update(
                {
                    "cpu_usage": f"""100 * sum(rate(node_cpu_seconds_total{{cluster="{cluster_name}",mode!="idle"}}[{rate_interval}])) by (instance) /
                            count by (instance) (node_cpu_seconds_total{{cluster="{cluster_name}"}})""",
                }
            )

        if metric_type in ["all", "disk"]:
            queries.update(
                {
                    "disk_total": f'sum by (instance, mountpoint) (node_filesystem_size_bytes{{cluster="{cluster_name}"}}) / 1024 / 1024 / 1024',
                    "disk_used": f'sum by (instance, mountpoint) ((node_filesystem_size_bytes{{cluster="{cluster_name}"}} - node_filesystem_free_bytes{{cluster="{cluster_name}"}})) / 1024 / 1024 / 1024',
                }
            )

        if metric_type in ["all", "network"]:
            queries.update(
                {
                    "network_receive": f'sum by (instance, device) (rate(node_network_receive_bytes_total{{cluster="{cluster_name}"}}[{rate_interval}]) * 8 / 1024 / 1024)',
                    "network_transmit": f'sum by (instance, device) (rate(node_network_transmit_bytes_total{{cluster="{cluster_name}"}}[{rate_interval}]) * 8 / 1024 / 1024)',
                    "network_errors": f'sum by (instance) (node_network_transmit_errs_total{{cluster="{cluster_name}"}} + node_network_receive_errs_total{{cluster="{cluster_name}"}}) or vector(0)',
                }
            )

        if metric_type in ["all", "gpu"]:
            queries.update(
                {
                    "gpu_memory_used": f'sum by (instance, gpu) (nvidia_gpu_memory_used_bytes{{cluster="{cluster_name}"}}) / 1024 / 1024 / 1024',
                    "gpu_memory_total": f'sum by (instance, gpu) (nvidia_gpu_memory_total_bytes{{cluster="{cluster_name}"}}) / 1024 / 1024 / 1024',
                    "gpu_utilization": f'avg by (instance, gpu) (nvidia_gpu_duty_cycle{{cluster="{cluster_name}"}}) or vector(0)',
                }
            )

        if metric_type in ["all", "hpu"]:
            queries.update(
                {
                    "hpu_memory_used": f'sum by (instance, hpu) (habana_memory_used_bytes{{cluster="{cluster_name}"}}) / 1024 / 1024 / 1024',
                    "hpu_memory_total": f'sum by (instance, hpu) (habana_memory_total_bytes{{cluster="{cluster_name}"}}) / 1024 / 1024 / 1024',
                    "hpu_utilization": f'avg by (instance, hpu) (habana_duty_cycle{{cluster="{cluster_name}"}}) or vector(0)',
                }
            )

        # Always include total nodes count regardless of metric type
        queries["total_nodes"] = f'count(node_uname_info{{cluster="{cluster_name}"}})'

        return queries

    async def get_cluster_metrics(
        self, cluster_id: UUID, time_range: str = "today", metric_type: str = "all"
    ) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive cluster metrics.

        Args:
            cluster_id: The cluster ID to fetch metrics for
            time_range: One of 'today', '7days', 'month'
            metric_type: Type of metrics to fetch ('all', 'memory', 'cpu', 'disk', 'gpu', 'hpu', 'network')

        Returns:
            Dict with node and summary metrics, filtered by metric_type
        """
        if not cluster_id:
            logger.error("Cluster ID is required to fetch cluster metrics")
            return None

        cluster_name = str(cluster_id)
        rate_interval = "5m" if time_range == "today" else "1h" if time_range == "7days" else "6h"

        # Get filtered queries based on metric type
        queries = self._get_metric_queries(cluster_name, rate_interval, metric_type)

        try:
            # Fetch metrics asynchronously
            results = await self._fetch_all_metrics(queries, time_range)

            if not any(result for result in results.values()):
                logger.error("No data returned from Prometheus queries")
                return None

            async with self._get_executor() as executor:
                # Process current values in parallel
                current_results = {
                    key: executor.submit(self._process_current_values, result)
                    for key, result in results.items()
                    if result
                }

                # Wait for all processing to complete
                current_results = {key: future.result() for key, future in current_results.items()}

                # Process node metrics
                nodes = self._process_node_metrics(current_results, time_range)
                current_results["nodes"] = nodes

                # Process cluster summary
                cluster_summary = self._process_cluster_summary(current_results, time_range)

                # Process historical data in parallel
                historical_data = self._process_historical_data(results, executor)

                metrics = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "time_range": time_range,
                    "historical_data": historical_data,
                    "nodes": nodes,
                    "cluster_summary": cluster_summary,
                }

            logger.info(f"Successfully fetched cluster metrics for time range: {time_range}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to fetch cluster metrics: {e}")
            return None

    def _process_current_values(self, result: List) -> List:
        """Process current values from time series data."""
        return [{"metric": series["metric"], "value": series["values"][-1]} for series in result]

    def _process_historical_data(self, results: Dict, executor: ThreadPoolExecutor) -> Dict:
        """Process historical data in parallel."""
        historical_futures = {
            key: executor.submit(self._process_series_historical_data, result)
            for key, result in results.items()
            if result
        }

        return {key: future.result() for key, future in historical_futures.items()}

    def _process_series_historical_data(self, result: List) -> List:
        """Process historical data for a single series."""
        return [{"timestamp": value[0], "value": float(value[1])} for series in result for value in series["values"]]

    def _get_unique_instances(self, results: Dict) -> List[str]:
        """Extract unique instance names from results."""
        instances = set()
        for result in results.values():
            for metric in result:
                if "instance" in metric["metric"]:
                    instances.add(metric["metric"]["instance"])
        return sorted(list(instances))

    def _get_instance_value(self, metrics: List, instance: str) -> float:
        """Get metric value for a specific instance."""
        for metric in metrics:
            if metric["metric"].get("instance") == instance:
                return metric["value"][1]
        return 0.0

    def _calculate_metric_change(self, current_value: float, historical_data: List, time_range: str) -> float:
        """Calculate the change in a metric based on the time range.
        
        Args:
            current_value: The current value of the metric
            historical_data: List of historical values for the metric
            time_range: One of 'today', '7days', 'month'
            
        Returns:
            Float representing the change in the metric (in the same units as the input)
        """
        if not historical_data or not len(historical_data):
            return 0.0
        
        try:
            # Get the first series of values
            first_series = historical_data[0]
            if not first_series.get("values"):
                return 0.0

            values = first_series["values"]
            # Get the first value (oldest) in the time series
            previous_value = float(values[0][1])
            return round(current_value - previous_value, 2)
            
        except (IndexError, ValueError, KeyError) as e:
            logger.error(f"Error calculating metric change: {e}")
            return 0.0

    def _process_node_metrics(self, results: Dict, time_range: str) -> Dict:
        """Process node metrics from results."""
        nodes = {}
        instances = self._get_unique_instances(results)

        for instance in instances:
            memory_total = float(self._get_instance_value(results.get("memory_total", []), instance))
            memory_available = float(self._get_instance_value(results.get("memory_available", []), instance))
            memory_used = memory_total - memory_available
            memory_usage_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0

            current_cpu_usage = float(self._get_instance_value(results.get("cpu_usage", []), instance))

            nodes[instance] = {
                "memory": {
                    "total_gib": memory_total,
                    "available_gib": memory_available,
                    "used_gib": memory_used,
                    "usage_percent": memory_usage_percent,
                    "change_percent": self._calculate_metric_change(
                        memory_usage_percent,
                        [m for m in results.get("memory_available", []) if m["metric"].get("instance") == instance],
                        time_range
                    )
                },
                "cpu": {
                    "cpu_usage_percent": current_cpu_usage,
                    "change_percent": self._calculate_metric_change(
                        current_cpu_usage,
                        [m for m in results.get("cpu_usage", []) if m["metric"].get("instance") == instance],
                        time_range
                    )
                },
                "disk": self._process_disk_metrics(results, instance, time_range),
                "network": self._process_network_metrics(results, instance, time_range),
                "gpu": self._process_gpu_metrics(results, instance, time_range),
                "hpu": self._process_hpu_metrics(results, instance, time_range),
            }

        return nodes

    def _process_disk_metrics(self, results: Dict, instance: str, time_range: str) -> Dict:
        """Process disk metrics for a specific instance."""
        disk_metrics = {}
        mountpoints = set()

        disk_total = results.get("disk_total", [])
        disk_used = results.get("disk_used", [])

        for metric in disk_total:
            if metric["metric"]["instance"] == instance:
                mountpoints.add(metric["metric"]["mountpoint"])

        for mountpoint in mountpoints:
            total = float(self._get_instance_mountpoint_value(disk_total, instance, mountpoint))
            used = float(self._get_instance_mountpoint_value(disk_used, instance, mountpoint))
            usage_percent = (used / total * 100) if total > 0 else 0

            disk_metrics[mountpoint] = {
                "total_gib": total,
                "used_gib": used,
                "available_gib": total - used,
                "usage_percent": usage_percent,
                "change_percent": self._calculate_metric_change(
                    usage_percent,
                    [m for m in disk_used if m["metric"].get("instance") == instance 
                     and m["metric"].get("mountpoint") == mountpoint],
                    time_range
                )
            }

        return disk_metrics

    def _process_network_metrics(self, results: Dict, instance: str, time_range: str) -> Dict:
        """Process network metrics for a specific instance."""
        interfaces = {}
        total_receive = 0
        total_transmit = 0

        # Safely get network metrics
        network_receive = results.get("network_receive", [])
        network_transmit = results.get("network_transmit", [])
        network_errors = results.get("network_errors", [])

        # Process per-interface metrics
        for metric in network_receive:
            if metric["metric"]["instance"] == instance:
                device = metric["metric"]["device"]
                receive = float(metric["value"][1])
                transmit = float(self._get_device_value(network_transmit, instance, device))

                interfaces[device] = {
                    "receive_mbps": receive,
                    "transmit_mbps": transmit,
                    "bandwidth_mbps": receive + transmit,
                }
                total_receive += receive
                total_transmit += transmit

        total_errors = float(self._get_instance_value(network_errors, instance))

        return {
            "interfaces": interfaces,
            "summary": {
                "total_receive_mbps": total_receive,
                "total_transmit_mbps": total_transmit,
                "total_bandwidth_mbps": total_receive + total_transmit,
                "total_errors": total_errors,
            },
            "change_percent": self._calculate_metric_change(
                total_errors,
                [m for m in network_errors if m["metric"].get("instance") == instance],
                time_range
            )
        }

    def _process_gpu_metrics(self, results: Dict, instance: str, time_range: str) -> Dict[str, float]:
        """Process GPU metrics for a specific instance."""
        gpu_memory_used = results.get("gpu_memory_used", [])
        gpu_memory_total = results.get("gpu_memory_total", [])
        gpu_utilization = results.get("gpu_utilization", [])

        try:
            return {
                "memory_used_gib": float(self._get_instance_value(gpu_memory_used, instance)),
                "memory_total_gib": float(self._get_instance_value(gpu_memory_total, instance)),
                "utilization_percent": float(self._get_instance_value(gpu_utilization, instance)),
                "change_percent": self._calculate_metric_change(
                    float(self._get_instance_value(gpu_utilization, instance)),
                    [m for m in gpu_utilization if m["metric"].get("instance") == instance],
                    time_range
                )
            }
        except (KeyError, TypeError, ValueError):
            return {"memory_used_gib": 0.0, "memory_total_gib": 0.0, "utilization_percent": 0.0}

    def _process_hpu_metrics(self, results: Dict, instance: str, time_range: str) -> Dict[str, float]:
        """Process HPU metrics for a specific instance."""
        hpu_memory_used = results.get("hpu_memory_used", [])
        hpu_memory_total = results.get("hpu_memory_total", [])
        hpu_utilization = results.get("hpu_utilization", [])

        try:
            return {
                "memory_used_gib": float(self._get_instance_value(hpu_memory_used, instance)),
                "memory_total_gib": float(self._get_instance_value(hpu_memory_total, instance)),
                "utilization_percent": float(self._get_instance_value(hpu_utilization, instance)),
                "change_percent": self._calculate_metric_change(
                    float(self._get_instance_value(hpu_utilization, instance)),
                    [m for m in hpu_utilization if m["metric"].get("instance") == instance],
                    time_range
                )
            }
        except (KeyError, TypeError, ValueError):
            return {"memory_used_gib": 0.0, "memory_total_gib": 0.0, "utilization_percent": 0.0}

    def _aggregate_disk_metrics(self, nodes: Dict) -> Dict[str, float]:
        """Aggregate disk metrics across all nodes."""
        total_size = 0.0
        total_used = 0.0

        for node in nodes.values():
            for mount_metrics in node["disk"].values():
                total_size += mount_metrics["total_gib"]
                total_used += mount_metrics["used_gib"]

        return {
            "total_gib": total_size,
            "used_gib": total_used,
            "available_gib": total_size - total_used,
            "usage_percent": (total_used / total_size * 100) if total_size > 0 else 0,
        }

    def _aggregate_network_metrics(self, nodes: Dict) -> Dict[str, float]:
        """Aggregate network metrics across all nodes."""
        total_receive = sum(node["network"]["summary"]["total_receive_mbps"] for node in nodes.values())
        total_transmit = sum(node["network"]["summary"]["total_transmit_mbps"] for node in nodes.values())
        total_errors = sum(node["network"]["summary"]["total_errors"] for node in nodes.values())

        return {
            "total_receive_mbps": total_receive,
            "total_transmit_mbps": total_transmit,
            "total_bandwidth_mbps": total_receive + total_transmit,
            "total_errors": total_errors,
        }

    def _aggregate_gpu_metrics(self, nodes: Dict) -> Dict[str, float]:
        """Aggregate GPU metrics across all nodes."""
        total_memory = sum(node["gpu"]["memory_total_gib"] for node in nodes.values())
        used_memory = sum(node["gpu"]["memory_used_gib"] for node in nodes.values())
        avg_utilization = (
            sum(node["gpu"]["utilization_percent"] for node in nodes.values()) / len(nodes) if nodes else 0.0
        )

        return {
            "memory_total_gib": total_memory,
            "memory_used_gib": used_memory,
            "memory_available_gib": total_memory - used_memory,
            "utilization_percent": avg_utilization,
            "memory_usage_percent": (used_memory / total_memory * 100) if total_memory > 0 else 0.0,
        }

    def _aggregate_hpu_metrics(self, nodes: Dict) -> Dict[str, float]:
        """Aggregate HPU metrics across all nodes."""
        total_memory = sum(node["hpu"]["memory_total_gib"] for node in nodes.values())
        used_memory = sum(node["hpu"]["memory_used_gib"] for node in nodes.values())
        avg_utilization = (
            sum(node["hpu"]["utilization_percent"] for node in nodes.values()) / len(nodes) if nodes else 0.0
        )

        return {
            "memory_total_gib": total_memory,
            "memory_used_gib": used_memory,
            "memory_available_gib": total_memory - used_memory,
            "utilization_percent": avg_utilization,
            "memory_usage_percent": (used_memory / total_memory * 100) if total_memory > 0 else 0.0,
        }

    def _get_instance_mountpoint_value(self, metrics: List, instance: str, mountpoint: str) -> float:
        """Get metric value for a specific instance and mountpoint."""
        try:
            for metric in metrics:
                if metric["metric"].get("instance") == instance and metric["metric"].get("mountpoint") == mountpoint:
                    return float(metric["value"][1])
        except (KeyError, TypeError, ValueError, IndexError):
            pass
        return 0.0

    def _get_device_value(self, metrics: List, instance: str, device: str) -> float:
        """Get metric value for a specific instance and network device."""
        try:
            for metric in metrics:
                if metric["metric"].get("instance") == instance and metric["metric"].get("device") == device:
                    return float(metric["value"][1])
        except (KeyError, TypeError, ValueError, IndexError):
            pass
        return 0.0

    def _process_cluster_summary(self, results: Dict, time_range: str) -> Dict:
        """Process cluster summary metrics from results."""
        nodes = results.get("nodes", {})
        if not nodes:
            return {
                "total_nodes": 0,
                "memory": {"total_gib": 0, "used_gib": 0, "available_gib": 0, "usage_percent": 0, "change_percent": 0},
                "cpu": {"average_usage_percent": 0, "change_percent": 0},
                "disk": {"total_gib": 0, "used_gib": 0, "available_gib": 0, "usage_percent": 0, "change_percent": 0},
                "network": {
                    "total_receive_mbps": 0,
                    "total_transmit_mbps": 0,
                    "total_bandwidth_mbps": 0,
                    "total_errors": 0,
                    "change_percent": 0
                },
                "gpu": {
                    "memory_total_gib": 0,
                    "memory_used_gib": 0,
                    "memory_available_gib": 0,
                    "utilization_percent": 0,
                    "memory_usage_percent": 0,
                    "change_percent": 0
                },
                "hpu": {
                    "memory_total_gib": 0,
                    "memory_used_gib": 0,
                    "memory_available_gib": 0,
                    "utilization_percent": 0,
                    "memory_usage_percent": 0,
                    "change_percent": 0
                },
            }

        # Calculate current values
        total_memory = sum(node["memory"]["total_gib"] for node in nodes.values())
        used_memory = sum(node["memory"]["used_gib"] for node in nodes.values())
        memory_usage_percent = (used_memory / total_memory * 100) if total_memory > 0 else 0
        
        current_cpu_usage = sum(node["cpu"]["cpu_usage_percent"] for node in nodes.values()) / len(nodes) if nodes else 0

        summary = {
            "total_nodes": int(self._get_instance_value(results.get("total_nodes", []), "total_nodes")),
            "memory": {
                "total_gib": total_memory,
                "used_gib": used_memory,
                "available_gib": total_memory - used_memory,
                "usage_percent": memory_usage_percent,
                "change_percent": self._calculate_metric_change(
                    memory_usage_percent,
                    results.get("memory_available", []),
                    time_range
                )
            },
            "cpu": {
                "average_usage_percent": current_cpu_usage,
                "change_percent": self._calculate_metric_change(
                    current_cpu_usage,
                    results.get("cpu_usage", []),
                    time_range
                )
            },
            "disk": self._aggregate_disk_metrics(nodes),
            "network": self._aggregate_network_metrics(nodes),
            "gpu": self._aggregate_gpu_metrics(nodes),
            "hpu": self._aggregate_hpu_metrics(nodes),
        }

        return summary
