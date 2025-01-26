import requests
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from budapp.commons import logging

logger = logging.get_logger(__name__)

class ClusterMetricsFetcher:
    """Fetches cluster metrics from Prometheus."""

    def __init__(self, prometheus_url: str):
        """Initialize with Prometheus server URL."""
        self.prometheus_url = prometheus_url
        self.api_url = f"{prometheus_url}/api/v1"

    def query(self, query: str) -> list:
        """Execute a Prometheus query.

        Args:
            query: The PromQL query string

        Returns:
            List of results from Prometheus

        Raises:
            requests.exceptions.RequestException: If query fails
        """
        try:
            response = requests.get(
                f"{self.api_url}/query",
                params={'query': query},
                verify=True,
                timeout=10
            )
            response.raise_for_status()
            return response.json()['data']['result']
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to execute Prometheus query: {e}")
            raise

    def get_cluster_metrics(self) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive cluster metrics.

        Returns:
            Dict containing various cluster metrics:
            - CPU usage and capacity
            - Memory usage and capacity
            - GPU metrics if available
            - Node status
            - Pod status

        Returns None if metrics cannot be fetched.
        """
        try:
            # CPU metrics
            cpu_usage = self.query('sum(rate(node_cpu_seconds_total{mode!="idle"}[5m])) by (instance)')
            cpu_capacity = self.query('sum(machine_cpu_cores) by (instance)')

            # Memory metrics
            memory_usage = self.query('sum(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) by (instance)')
            memory_capacity = self.query('sum(node_memory_MemTotal_bytes) by (instance)')

            # GPU metrics if available
            gpu_metrics = {}
            try:
                gpu_usage = self.query('sum(nvidia_gpu_duty_cycle) by (gpu)')
                gpu_memory = self.query('sum(nvidia_gpu_memory_used_bytes) by (gpu)')
                gpu_metrics = {
                    'gpu_utilization': self._process_gpu_metrics(gpu_usage),
                    'gpu_memory': self._process_gpu_metrics(gpu_memory)
                }
            except Exception as e:
                logger.warning(f"GPU metrics not available: {e}")

            # Node status
            nodes_ready = self.query('sum(kube_node_status_condition{condition="Ready",status="true"})')
            nodes_not_ready = self.query('sum(kube_node_status_condition{condition="Ready",status="false"})')

            # Pod status
            pods_running = self.query('sum(kube_pod_status_phase{phase="Running"})')
            pods_failed = self.query('sum(kube_pod_status_phase{phase="Failed"})')
            pods_pending = self.query('sum(kube_pod_status_phase{phase="Pending"})')

            # Network metrics
            network_receive = self.query('sum(rate(node_network_receive_bytes_total[5m])) by (instance)')
            network_transmit = self.query('sum(rate(node_network_transmit_bytes_total[5m])) by (instance)')

            # Disk metrics
            disk_usage = self.query('sum(node_filesystem_size_bytes{mountpoint="/"} - node_filesystem_free_bytes{mountpoint="/"}) by (instance)')
            disk_capacity = self.query('sum(node_filesystem_size_bytes{mountpoint="/"}) by (instance)')

            metrics = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cpu': {
                    'usage': self._process_cpu_metrics(cpu_usage),
                    'capacity': self._process_cpu_metrics(cpu_capacity),
                    'utilization': self._calculate_utilization(cpu_usage, cpu_capacity)
                },
                'memory': {
                    'usage': self._process_memory_metrics(memory_usage),
                    'capacity': self._process_memory_metrics(memory_capacity),
                    'utilization': self._calculate_utilization(memory_usage, memory_capacity)
                },
                'gpu': gpu_metrics if gpu_metrics else None,
                'nodes': {
                    'ready': self._process_scalar_metric(nodes_ready),
                    'not_ready': self._process_scalar_metric(nodes_not_ready)
                },
                'pods': {
                    'running': self._process_scalar_metric(pods_running),
                    'failed': self._process_scalar_metric(pods_failed),
                    'pending': self._process_scalar_metric(pods_pending)
                },
                'network': {
                    'receive_bytes_per_second': self._process_network_metrics(network_receive),
                    'transmit_bytes_per_second': self._process_network_metrics(network_transmit)
                },
                'disk': {
                    'usage': self._process_disk_metrics(disk_usage),
                    'capacity': self._process_disk_metrics(disk_capacity),
                    'utilization': self._calculate_utilization(disk_usage, disk_capacity)
                }
            }

            logger.info("Successfully fetched cluster metrics")
            logger.debug(f"XXX Cluster metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to fetch cluster metrics: {e}")
            return None

    def _process_cpu_metrics(self, metric_data: list) -> float:
        """Process CPU metrics from raw Prometheus data."""
        try:
            if not metric_data:
                return 0.0
            return float(metric_data[0]['value'][1])
        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"Error processing CPU metrics: {e}")
            return 0.0

    def _process_memory_metrics(self, metric_data: list) -> float:
        """Process memory metrics from raw Prometheus data."""
        try:
            if not metric_data:
                return 0.0
            # Convert bytes to GB
            return float(metric_data[0]['value'][1]) / (1024 * 1024 * 1024)
        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"Error processing memory metrics: {e}")
            return 0.0

    def _process_gpu_metrics(self, metric_data: list) -> Dict[str, float]:
        """Process GPU metrics from raw Prometheus data."""
        try:
            if not metric_data:
                return {}
            return {
                item['metric']['gpu']: float(item['value'][1])
                for item in metric_data
            }
        except (KeyError, ValueError) as e:
            logger.error(f"Error processing GPU metrics: {e}")
            return {}

    def _process_scalar_metric(self, metric_data: list) -> int:
        """Process scalar metrics from raw Prometheus data."""
        try:
            if not metric_data:
                return 0
            return int(float(metric_data[0]['value'][1]))
        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"Error processing scalar metric: {e}")
            return 0

    def _process_network_metrics(self, metric_data: list) -> Dict[str, float]:
        """Process network metrics from raw Prometheus data."""
        try:
            if not metric_data:
                return {}
            return {
                item['metric']['instance']: float(item['value'][1])
                for item in metric_data
            }
        except (KeyError, ValueError) as e:
            logger.error(f"Error processing network metrics: {e}")
            return {}

    def _process_disk_metrics(self, metric_data: list) -> float:
        """Process disk metrics from raw Prometheus data."""
        try:
            if not metric_data:
                return 0.0
            # Convert bytes to GB
            return float(metric_data[0]['value'][1]) / (1024 * 1024 * 1024)
        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"Error processing disk metrics: {e}")
            return 0.0

    def _calculate_utilization(self, usage_data: list, capacity_data: list) -> float:
        """Calculate utilization percentage from usage and capacity metrics."""
        try:
            if not usage_data or not capacity_data:
                return 0.0
            usage = float(usage_data[0]['value'][1])
            capacity = float(capacity_data[0]['value'][1])
            if capacity == 0:
                return 0.0
            return (usage / capacity) * 100
        except (IndexError, KeyError, ValueError, ZeroDivisionError) as e:
            logger.error(f"Error calculating utilization: {e}")
            return 0.0
