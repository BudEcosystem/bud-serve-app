import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from budapp.commons import logging
from uuid import UUID

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

    def get_cluster_metrics(self, cluster_id: UUID) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive cluster metrics.

        Args:
            cluster_id: UUID of the cluster to fetch metrics for

        Returns:
            Dict containing cluster metrics structured according to ClusterNodeMetrics
            and ClusterSummaryMetrics models
        """
        if not cluster_id:
            logger.error("Cluster ID is required to fetch cluster metrics")
            return None

        cluster_name = cluster_id  # This should be dynamic based on cluster_id

        # Define all Prometheus queries
        queries = {
            # Per-node memory metrics
            'memory_total': f'node_memory_MemTotal_bytes{{cluster="{cluster_name}"}} / 1024 / 1024 / 1024',
            'memory_available': f'node_memory_MemAvailable_bytes{{cluster="{cluster_name}"}} / 1024 / 1024 / 1024',

            # Per-node CPU metrics
            'cpu_usage': f'''100 * sum(rate(node_cpu_seconds_total{{cluster="{cluster_name}",mode!="idle"}}[5m])) by (instance) /
                        count by (instance) (node_cpu_seconds_total{{cluster="{cluster_name}"}})''',

            # Per-node disk metrics
            'disk_total': f'sum by (instance, mountpoint) (node_filesystem_size_bytes{{cluster="{cluster_name}"}}) / 1024 / 1024 / 1024',
            'disk_used': f'sum by (instance, mountpoint) ((node_filesystem_size_bytes{{cluster="{cluster_name}"}} - node_filesystem_free_bytes{{cluster="{cluster_name}"}})) / 1024 / 1024 / 1024',

            # Per-node network metrics
            'network_receive': f'sum by (instance, device) (rate(node_network_receive_bytes_total{{cluster="{cluster_name}"}}[5m]) * 8 / 1024 / 1024)',
            'network_transmit': f'sum by (instance, device) (rate(node_network_transmit_bytes_total{{cluster="{cluster_name}"}}[5m]) * 8 / 1024 / 1024)',
            'network_errors': f'sum by (instance) (node_network_transmit_errs_total{{cluster="{cluster_name}"}} + node_network_receive_errs_total{{cluster="{cluster_name}"}}) or vector(0)',

            # GPU metrics
            'gpu_memory_used': f'sum by (instance, gpu) (nvidia_gpu_memory_used_bytes{{cluster="{cluster_name}"}}) / 1024 / 1024 / 1024',
            'gpu_memory_total': f'sum by (instance, gpu) (nvidia_gpu_memory_total_bytes{{cluster="{cluster_name}"}}) / 1024 / 1024 / 1024',
            'gpu_utilization': f'avg by (instance, gpu) (nvidia_gpu_duty_cycle{{cluster="{cluster_name}"}}) or vector(0)',

            # HPU metrics if available
            'hpu_memory_used': f'sum by (instance, hpu) (habana_memory_used_bytes{{cluster="{cluster_name}"}}) / 1024 / 1024 / 1024',
            'hpu_memory_total': f'sum by (instance, hpu) (habana_memory_total_bytes{{cluster="{cluster_name}"}}) / 1024 / 1024 / 1024',
            'hpu_utilization': f'avg by (instance, hpu) (habana_duty_cycle{{cluster="{cluster_name}"}}) or vector(0)',

            # Node count
            'total_nodes': f'count(node_uname_info{{cluster="{cluster_name}"}})'
        }

        try:
            results = {key: self.query(query) for key, query in queries.items()}

            # Process per-node metrics
            nodes = {}
            instances = self._get_unique_instances(results)

            for instance in instances:
                nodes[instance] = {
                    'memory': {
                        'total_gib': float(self._get_instance_value(results['memory_total'], instance)),
                        'available_gib': float(self._get_instance_value(results['memory_available'], instance)),
                        'used_gib': float(self._get_instance_value(results['memory_total'], instance)) -
                                   float(self._get_instance_value(results['memory_available'], instance)),
                        'usage_percent': (1 - float(self._get_instance_value(results['memory_available'], instance)) /
                                        float(self._get_instance_value(results['memory_total'], instance))) * 100
                    },
                    'cpu': {
                        'cpu_usage_percent': float(self._get_instance_value(results['cpu_usage'], instance))
                    },
                    'disk': self._process_disk_metrics(results, instance),
                    'network': self._process_network_metrics(results, instance),
                    'gpu': self._process_gpu_metrics(results, instance),
                    'hpu': self._process_hpu_metrics(results, instance)
                }

            # Process cluster summary metrics
            summary = {
                'total_nodes': int(results['total_nodes'][0]['value'][1]),
                'memory': {
                    'total_gib': sum(node['memory']['total_gib'] for node in nodes.values()),
                    'used_gib': sum(node['memory']['used_gib'] for node in nodes.values()),
                    'available_gib': sum(node['memory']['available_gib'] for node in nodes.values()),
                    'usage_percent': sum(node['memory']['used_gib'] for node in nodes.values()) /
                                   sum(node['memory']['total_gib'] for node in nodes.values()) * 100
                },
                'cpu': {
                    'average_usage_percent': sum(node['cpu']['cpu_usage_percent'] for node in nodes.values()) / len(nodes)
                },
                'disk': self._aggregate_disk_metrics(nodes),
                'network': self._aggregate_network_metrics(nodes),
                'gpu': self._aggregate_gpu_metrics(nodes),
                'hpu': self._aggregate_hpu_metrics(nodes)
            }

            metrics = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'nodes': nodes,
                'summary': summary
            }

            logger.info("Successfully fetched cluster metrics")
            logger.debug(metrics)
            return metrics

        except Exception as e:
            logger.error(f"Failed to fetch cluster metrics: {e}")
            return None

    def _get_unique_instances(self, results: Dict) -> List[str]:
        """Extract unique instance names from results."""
        instances = set()
        for result in results.values():
            for metric in result:
                if 'instance' in metric['metric']:
                    instances.add(metric['metric']['instance'])
        return sorted(list(instances))

    def _get_instance_value(self, metrics: List, instance: str) -> float:
        """Get metric value for a specific instance."""
        for metric in metrics:
            if metric['metric'].get('instance') == instance:
                return metric['value'][1]
        return 0.0

    def _process_disk_metrics(self, results: Dict, instance: str) -> Dict:
        """Process disk metrics for a specific instance."""
        disk_metrics = {}
        mountpoints = set()

        # Get unique mountpoints for this instance
        for metric in results['disk_total']:
            if metric['metric']['instance'] == instance:
                mountpoints.add(metric['metric']['mountpoint'])

        for mountpoint in mountpoints:
            total = float(self._get_instance_mountpoint_value(results['disk_total'], instance, mountpoint))
            used = float(self._get_instance_mountpoint_value(results['disk_used'], instance, mountpoint))
            disk_metrics[mountpoint] = {
                'total_gib': total,
                'used_gib': used,
                'available_gib': total - used,
                'usage_percent': (used / total * 100) if total > 0 else 0
            }

        return disk_metrics

    def _process_network_metrics(self, results: Dict, instance: str) -> Dict:
        """Process network metrics for a specific instance."""
        interfaces = {}
        total_receive = 0
        total_transmit = 0

        # Process per-interface metrics
        for metric in results['network_receive']:
            if metric['metric']['instance'] == instance:
                device = metric['metric']['device']
                receive = float(metric['value'][1])
                transmit = float(self._get_device_value(results['network_transmit'], instance, device))

                interfaces[device] = {
                    'receive_mbps': receive,
                    'transmit_mbps': transmit,
                    'bandwidth_mbps': receive + transmit
                }
                total_receive += receive
                total_transmit += transmit

        total_errors = float(self._get_instance_value(results['network_errors'], instance))

        return {
            'interfaces': interfaces,
            'summary': {
                'total_receive_mbps': total_receive,
                'total_transmit_mbps': total_transmit,
                'total_bandwidth_mbps': total_receive + total_transmit,
                'total_errors': total_errors
            }
        }

    def _process_gpu_metrics(self, results: Dict, instance: str) -> Dict[str, float]:
        """Process GPU metrics for a specific instance."""
        try:
            return {
                'memory_used_gib': float(self._get_instance_value(results['gpu_memory_used'], instance)),
                'memory_total_gib': float(self._get_instance_value(results['gpu_memory_total'], instance)),
                'utilization_percent': float(self._get_instance_value(results['gpu_utilization'], instance))
            }
        except (KeyError, TypeError, ValueError):
            return {
                'memory_used_gib': 0.0,
                'memory_total_gib': 0.0,
                'utilization_percent': 0.0
            }

    def _process_hpu_metrics(self, results: Dict, instance: str) -> Dict[str, float]:
        """Process HPU metrics for a specific instance."""
        try:
            return {
                'memory_used_gib': float(self._get_instance_value(results['hpu_memory_used'], instance)),
                'memory_total_gib': float(self._get_instance_value(results['hpu_memory_total'], instance)),
                'utilization_percent': float(self._get_instance_value(results['hpu_utilization'], instance))
            }
        except (KeyError, TypeError, ValueError):
            return {
                'memory_used_gib': 0.0,
                'memory_total_gib': 0.0,
                'utilization_percent': 0.0
            }

    def _aggregate_disk_metrics(self, nodes: Dict) -> Dict[str, float]:
        """Aggregate disk metrics across all nodes."""
        total_size = 0.0
        total_used = 0.0

        for node in nodes.values():
            for mount_metrics in node['disk'].values():
                total_size += mount_metrics['total_gib']
                total_used += mount_metrics['used_gib']

        return {
            'total_gib': total_size,
            'used_gib': total_used,
            'available_gib': total_size - total_used,
            'usage_percent': (total_used / total_size * 100) if total_size > 0 else 0
        }

    def _aggregate_network_metrics(self, nodes: Dict) -> Dict[str, float]:
        """Aggregate network metrics across all nodes."""
        total_receive = sum(node['network']['summary']['total_receive_mbps'] for node in nodes.values())
        total_transmit = sum(node['network']['summary']['total_transmit_mbps'] for node in nodes.values())
        total_errors = sum(node['network']['summary']['total_errors'] for node in nodes.values())

        return {
            'total_receive_mbps': total_receive,
            'total_transmit_mbps': total_transmit,
            'total_bandwidth_mbps': total_receive + total_transmit,
            'total_errors': total_errors
        }

    def _aggregate_gpu_metrics(self, nodes: Dict) -> Dict[str, float]:
        """Aggregate GPU metrics across all nodes."""
        total_memory = sum(node['gpu']['memory_total_gib'] for node in nodes.values())
        used_memory = sum(node['gpu']['memory_used_gib'] for node in nodes.values())
        avg_utilization = sum(node['gpu']['utilization_percent'] for node in nodes.values()) / len(nodes) if nodes else 0.0

        return {
            'memory_total_gib': total_memory,
            'memory_used_gib': used_memory,
            'memory_available_gib': total_memory - used_memory,
            'utilization_percent': avg_utilization,
            'memory_usage_percent': (used_memory / total_memory * 100) if total_memory > 0 else 0.0
        }

    def _aggregate_hpu_metrics(self, nodes: Dict) -> Dict[str, float]:
        """Aggregate HPU metrics across all nodes."""
        total_memory = sum(node['hpu']['memory_total_gib'] for node in nodes.values())
        used_memory = sum(node['hpu']['memory_used_gib'] for node in nodes.values())
        avg_utilization = sum(node['hpu']['utilization_percent'] for node in nodes.values()) / len(nodes) if nodes else 0.0

        return {
            'memory_total_gib': total_memory,
            'memory_used_gib': used_memory,
            'memory_available_gib': total_memory - used_memory,
            'utilization_percent': avg_utilization,
            'memory_usage_percent': (used_memory / total_memory * 100) if total_memory > 0 else 0.0
        }

    def _get_instance_mountpoint_value(self, metrics: List, instance: str, mountpoint: str) -> float:
        """Get metric value for a specific instance and mountpoint."""
        for metric in metrics:
            if (metric['metric'].get('instance') == instance and
                metric['metric'].get('mountpoint') == mountpoint):
                return metric['value'][1]
        return 0.0

    def _get_device_value(self, metrics: List, instance: str, device: str) -> float:
        """Get metric value for a specific instance and network device."""
        for metric in metrics:
            if (metric['metric'].get('instance') == instance and
                metric['metric'].get('device') == device):
                return metric['value'][1]
        return 0.0
