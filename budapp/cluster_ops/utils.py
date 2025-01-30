import json
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
            time_range: One of '10min', 'today', '7days', 'month'

        Returns:
            Dict with start and end timestamps and step interval
        """
        now = datetime.now(timezone.utc)

        if time_range == "10min":
            start_time = now - timedelta(minutes=10)
            step = "30s"  # 30-second intervals for 10 minutes
        elif time_range == "today":
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            step = "5m"  # 5-minute intervals for today
        elif time_range == "7days":
            start_time = now - timedelta(days=7)
            step = "1h"  # 1-hour intervals for 7 days
        elif time_range == "month":
            start_time = now - timedelta(days=30)
            step = "6h"  # 6-hour intervals for monthly view
        else:
            # Default to 10min if invalid range
            start_time = now - timedelta(minutes=10)
            step = "30s"

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

    def _calculate_metric_changes(self, current_metrics: Dict, previous_metrics: Dict) -> Dict:
        """Calculate changes between current and previous metrics.

        Args:
            current_metrics: Current period metrics
            previous_metrics: Previous period metrics

        Returns:
            Dict containing the changes in metrics
        """
        changes = {}

        try:
            logger.debug(f"Current metrics: {current_metrics}")
            logger.debug(f"Previous metrics: {previous_metrics}")
            # CPU change
            current_cpu = current_metrics.get("cpu", {}).get("average_usage_percent", 0)
            previous_cpu = previous_metrics.get("cpu", {}).get("average_usage_percent", 0)

            logger.debug(f"CPU values - Current: {current_cpu}, Previous: {previous_cpu}")
            changes["cpu"] = {
                "change": round(current_cpu - previous_cpu, 2),
                "change_percent": round(
                    ((current_cpu - previous_cpu) / previous_cpu * 100) if previous_cpu != 0 else 0, 2
                ),
            }

            # Memory change
            current_mem = current_metrics.get("memory", {}).get("usage_percent", 0)
            previous_mem = previous_metrics.get("memory", {}).get("usage_percent", 0)
            changes["memory"] = {
                "change": round(current_mem - previous_mem, 2),
                "change_percent": round(
                    ((current_mem - previous_mem) / previous_mem * 100) if previous_mem != 0 else 0, 2
                ),
            }

            # GPU utilization change
            current_gpu = current_metrics.get("gpu", {}).get("utilization_percent", 0)
            previous_gpu = previous_metrics.get("gpu", {}).get("utilization_percent", 0)
            changes["gpu"] = {
                "change": round(current_gpu - previous_gpu, 2),
                "change_percent": round(
                    ((current_gpu - previous_gpu) / previous_gpu * 100) if previous_gpu != 0 else 0, 2
                ),
            }

            # HPU utilization change
            current_hpu = current_metrics.get("hpu", {}).get("utilization_percent", 0)
            previous_hpu = previous_metrics.get("hpu", {}).get("utilization_percent", 0)
            changes["hpu"] = {
                "change": round(current_hpu - previous_hpu, 2),
                "change_percent": round(
                    ((current_hpu - previous_hpu) / previous_hpu * 100) if previous_hpu != 0 else 0, 2
                ),
            }

            # Network bandwidth change
            current_net = current_metrics.get("network", {}).get("total_bandwidth_mbps", 0)
            previous_net = previous_metrics.get("network", {}).get("total_bandwidth_mbps", 0)
            changes["network"] = {
                "change": round(current_net - previous_net, 2),
                "change_percent": round(
                    ((current_net - previous_net) / previous_net * 100) if previous_net != 0 else 0, 2
                ),
            }

            # Disk usage change
            current_disk = current_metrics.get("disk", {}).get("usage_percent", 0)
            previous_disk = previous_metrics.get("disk", {}).get("usage_percent", 0)
            changes["disk"] = {
                "change": round(current_disk - previous_disk, 2),
                "change_percent": round(
                    ((current_disk - previous_disk) / previous_disk * 100) if previous_disk != 0 else 0, 2
                ),
            }

        except Exception as e:
            logger.error(f"Error calculating metric changes: {e}")
            return {}

        return changes

    async def get_cluster_metrics(
        self, cluster_id: UUID, time_range: str = "today", metric_type: str = "all"
    ) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive cluster metrics.

        Args:
            cluster_id: The cluster ID to fetch metrics for
            time_range: One of '10min', 'today', '7days', 'month'
            metric_type: Type of metrics to fetch ('all', 'memory', 'cpu', 'disk', 'gpu', 'hpu', 'network')

        Returns:
            Dict with node and summary metrics, filtered by metric_type
        """
        if not cluster_id:
            logger.error("Cluster ID is required to fetch cluster metrics")
            return None

        cluster_name = str(cluster_id)
        rate_interval = (
            "1m"
            if time_range == "10min"
            else "5m"
            if time_range == "today"
            else "1h"
            if time_range == "7days"
            else "6h"
        )

        # Get filtered queries based on metric type
        queries = self._get_metric_queries(cluster_name, rate_interval, metric_type)

        try:
            # Fetch current period metrics
            results = await self._fetch_all_metrics(queries, time_range)

            if not any(result for result in results.values()):
                logger.error("No data returned from Prometheus queries")
                return None

            # Calculate the previous time range and get previous metrics
            now = datetime.now(timezone.utc)
            if time_range == "10min":
                previous_end = now - timedelta(minutes=10)  # End 10 minutes ago
                previous_start = previous_end - timedelta(minutes=10)  # Start 20 minutes ago
                previous_time_params = {
                    "start": previous_start.timestamp(),
                    "end": previous_end.timestamp(),
                    "step": "30s",
                }
            elif time_range == "today":
                previous_end = now - timedelta(days=1)  # End at yesterday
                previous_start = previous_end.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )  # Start at beginning of yesterday
                previous_time_params = {
                    "start": previous_start.timestamp(),
                    "end": previous_end.timestamp(),
                    "step": "5m",
                }
            elif time_range == "7days":
                previous_end = now - timedelta(days=7)  # End 7 days ago
                previous_start = previous_end - timedelta(days=7)  # Start 14 days ago
                previous_time_params = {
                    "start": previous_start.timestamp(),
                    "end": previous_end.timestamp(),
                    "step": "1h",
                }
            else:  # month
                previous_end = now - timedelta(days=30)  # End 30 days ago
                previous_start = previous_end - timedelta(days=30)  # Start 60 days ago
                previous_time_params = {
                    "start": previous_start.timestamp(),
                    "end": previous_end.timestamp(),
                    "step": "6h",
                }

            # Fetch previous period metrics with correct time range
            async with aiohttp.ClientSession() as session:
                previous_tasks = []
                for key, query in queries.items():
                    task = asyncio.create_task(self._async_query_range(session, query, previous_time_params))
                    previous_tasks.append((key, task))

                previous_results = {}
                for key, task in previous_tasks:
                    try:
                        previous_results[key] = await task
                    except Exception as e:
                        logger.error(f"Failed to fetch previous metrics for {key}: {e}")
                        previous_results[key] = []

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
                nodes = self._process_node_metrics(current_results)

                current_results["nodes"] = nodes

                # Process cluster summary for both current and previous periods
                current_summary = self._process_cluster_summary(current_results)

                # Process previous period results
                previous_processed = {
                    key: executor.submit(self._process_current_values, result).result()
                    for key, result in previous_results.items()
                    if result
                }

                previous_nodes = self._process_node_metrics(previous_processed)
                previous_processed["nodes"] = previous_nodes

                previous_summary = self._process_cluster_summary(previous_processed)

                # Calculate changes
                changes = self._calculate_metric_changes(current_summary, previous_summary)
                current_summary["changes"] = changes

                # Process historical data in parallel
                historical_data = self._process_historical_data(results, executor)

                metrics = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "time_range": time_range,
                    "historical_data": historical_data,
                    "nodes": nodes,
                    "cluster_summary": current_summary,
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

    def _process_disk_metrics(self, results: Dict, instance: str) -> Dict:
        """Process disk metrics for a specific instance."""
        disk_metrics = {}
        mountpoints = set()

        # Safely get disk metrics
        disk_total = results.get("disk_total", [])
        disk_used = results.get("disk_used", [])

        # Get unique mountpoints for this instance
        for metric in disk_total:
            if metric["metric"]["instance"] == instance:
                mountpoints.add(metric["metric"]["mountpoint"])

        for mountpoint in mountpoints:
            total = float(self._get_instance_mountpoint_value(disk_total, instance, mountpoint))
            used = float(self._get_instance_mountpoint_value(disk_used, instance, mountpoint))
            disk_metrics[mountpoint] = {
                "total_gib": total,
                "used_gib": used,
                "available_gib": total - used,
                "usage_percent": (used / total * 100) if total > 0 else 0,
            }

        return disk_metrics

    def _process_network_metrics(self, results: Dict, instance: str) -> Dict:
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
        }

    def _process_gpu_metrics(self, results: Dict, instance: str) -> Dict[str, float]:
        """Process GPU metrics for a specific instance."""
        gpu_memory_used = results.get("gpu_memory_used", [])
        gpu_memory_total = results.get("gpu_memory_total", [])
        gpu_utilization = results.get("gpu_utilization", [])

        try:
            return {
                "memory_used_gib": float(self._get_instance_value(gpu_memory_used, instance)),
                "memory_total_gib": float(self._get_instance_value(gpu_memory_total, instance)),
                "utilization_percent": float(self._get_instance_value(gpu_utilization, instance)),
            }
        except (KeyError, TypeError, ValueError):
            return {"memory_used_gib": 0.0, "memory_total_gib": 0.0, "utilization_percent": 0.0}

    def _process_hpu_metrics(self, results: Dict, instance: str) -> Dict[str, float]:
        """Process HPU metrics for a specific instance."""
        hpu_memory_used = results.get("hpu_memory_used", [])
        hpu_memory_total = results.get("hpu_memory_total", [])
        hpu_utilization = results.get("hpu_utilization", [])

        try:
            return {
                "memory_used_gib": float(self._get_instance_value(hpu_memory_used, instance)),
                "memory_total_gib": float(self._get_instance_value(hpu_memory_total, instance)),
                "utilization_percent": float(self._get_instance_value(hpu_utilization, instance)),
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

    def _process_node_metrics(self, results: Dict) -> Dict:
        """Process node metrics from results."""
        nodes = {}
        instances = self._get_unique_instances(results)

        for instance in instances:
            # Default values in case metrics are missing
            memory_total = float(self._get_instance_value(results.get("memory_total", []), instance))
            memory_available = float(self._get_instance_value(results.get("memory_available", []), instance))

            # Avoid division by zero
            memory_usage_percent = 0
            if memory_total > 0:
                memory_usage_percent = (1 - memory_available / memory_total) * 100

            nodes[instance] = {
                "memory": {
                    "total_gib": memory_total,
                    "available_gib": memory_available,
                    "used_gib": memory_total - memory_available,
                    "usage_percent": memory_usage_percent,
                },
                "cpu": {"cpu_usage_percent": float(self._get_instance_value(results.get("cpu_usage", []), instance))},
                "disk": self._process_disk_metrics(results, instance),
                "network": self._process_network_metrics(results, instance),
                "gpu": self._process_gpu_metrics(results, instance),
                "hpu": self._process_hpu_metrics(results, instance),
            }

        return nodes

    def _process_cluster_summary(self, results: Dict) -> Dict:
        """Process cluster summary metrics from results."""
        nodes = results.get("nodes", {})

        if not nodes:
            # Return default values if no node data is available
            return {
                "total_nodes": 0,
                "memory": {"total_gib": 0, "used_gib": 0, "available_gib": 0, "usage_percent": 0},
                "cpu": {"average_usage_percent": 0},
                "disk": {"total_gib": 0, "used_gib": 0, "available_gib": 0, "usage_percent": 0},
                "network": {
                    "total_receive_mbps": 0,
                    "total_transmit_mbps": 0,
                    "total_bandwidth_mbps": 0,
                    "total_errors": 0,
                },
                "gpu": {
                    "memory_total_gib": 0,
                    "memory_used_gib": 0,
                    "memory_available_gib": 0,
                    "utilization_percent": 0,
                    "memory_usage_percent": 0,
                },
                "hpu": {
                    "memory_total_gib": 0,
                    "memory_used_gib": 0,
                    "memory_available_gib": 0,
                    "utilization_percent": 0,
                    "memory_usage_percent": 0,
                },
            }

        total_memory = sum(node["memory"]["total_gib"] for node in nodes.values())
        used_memory = sum(node["memory"]["used_gib"] for node in nodes.values())

        summary = {
            "total_nodes": int(self._get_instance_value(results.get("total_nodes", []), "total_nodes")),
            "memory": {
                "total_gib": total_memory,
                "used_gib": used_memory,
                "available_gib": total_memory - used_memory,
                "usage_percent": (used_memory / total_memory * 100) if total_memory > 0 else 0,
            },
            "cpu": {
                "average_usage_percent": sum(node["cpu"]["cpu_usage_percent"] for node in nodes.values()) / len(nodes)
                if nodes
                else 0
            },
            "disk": self._aggregate_disk_metrics(nodes),
            "network": self._aggregate_network_metrics(nodes),
            "gpu": self._aggregate_gpu_metrics(nodes),
            "hpu": self._aggregate_hpu_metrics(nodes),
        }

        return summary
