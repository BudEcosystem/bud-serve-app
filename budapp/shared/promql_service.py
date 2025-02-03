from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import aiohttp
import asyncio
from cachetools import TTLCache
from functools import lru_cache
import urllib.parse
import json
from ..commons import logging
from ..commons.config import app_settings

logger = logging.get_logger(__name__)

class PromQLService:
    """Improved PromQL service class with async support and caching."""
    def _parse_time_input(self, time_input: Union[str, datetime]) -> str:
        """Convert various time inputs to Unix timestamp string.
        
        Args:
            time_input: Can be:
                - datetime object
                - "now"
                - relative time string (e.g., "1h", "5m", "2d")
                - Unix timestamp string
        
        Returns:
            Unix timestamp as string
        """
        if isinstance(time_input, datetime):
            return str(int(time_input.timestamp()))
        
        if time_input == "now":
            return str(int(datetime.now().timestamp()))
        
        # Handle relative time strings
        if isinstance(time_input, str):
            # Parse relative time strings like "1h", "5m", etc.
            time_units = {
                'h': 'hours',
                'm': 'minutes',
                'd': 'days',
                'w': 'weeks'
            }
            
            if any(time_input.endswith(unit) for unit in time_units.keys()):
                try:
                    value = int(time_input[:-1])
                    unit = time_input[-1]
                    if unit in time_units:
                        delta = timedelta(**{time_units[unit]: value})
                        return str(int((datetime.now() - delta).timestamp()))
                except ValueError:
                    pass
            
            # If it's already a timestamp string, return as is
            try:
                int(time_input)
                return time_input
            except ValueError:
                pass
                
        raise ValueError(f"Invalid time format: {time_input}")

    def __init__(self, cluster_name: str):
        """Initialize the PromQL service.
        
        Args:
            cluster_name (str): Name of the cluster to filter metrics
        """
        self.prometheus_url = app_settings.prometheus_url
        self.api_url = f"{self.prometheus_url}"
        self.cluster_name = cluster_name
        self.cluster_label = f'cluster="{self.cluster_name}"'
        
        # Initialize caches
        self._query_cache = TTLCache(maxsize=100, ttl=60)  # 1-minute cache for instant queries
        self._range_cache = TTLCache(maxsize=50, ttl=300)  # 5-minute cache for range queries
        
        # Initialize aiohttp session
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _close_session(self):
        """Close aiohttp session if it exists."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _add_cluster_filter(self, query: str) -> str:
        """
        Add cluster filter to a PromQL query.
        Ensures cluster label is only added once and maintains proper label syntax.
        """
        # Check if cluster label already exists in query
        if self.cluster_label in query:
            return query

        # Find all label groups (parts between curly braces)
        start_idx = 0
        brace_count = 0
        in_label_group = False
        label_groups = []
        current_group_start = 0

        for i, char in enumerate(query):
            if char == '{':
                brace_count += 1
                if brace_count == 1:
                    in_label_group = True
                    current_group_start = i
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and in_label_group:
                    in_label_group = False
                    label_groups.append((current_group_start, i))

        if not label_groups:
            # No existing label groups, add cluster label at the end
            return f"{query}{{{self.cluster_label}}}"

        # Add cluster label to the last label group
        last_group_start, last_group_end = label_groups[-1]
        label_content = query[last_group_start + 1:last_group_end]
        
        # Add cluster label with proper comma handling
        if label_content.strip():
            new_label_content = f"{label_content},{self.cluster_label}"
        else:
            new_label_content = self.cluster_label

        # Reconstruct the query
        return (
            query[:last_group_start + 1] +  # Everything before the last label group
            new_label_content +              # Modified label content
            query[last_group_end:]          # Everything after the last label group
        )

    @lru_cache(maxsize=100)
    def _build_cache_key(self, query: str, **params) -> str:
        """Build a cache key from query and parameters."""
        return f"{query}:{json.dumps(params, sort_keys=True)}"

    def _validate_query(self, query: str) -> None:
        """
        Validate a PromQL query before execution.
        Raises ValueError if query is invalid.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        # Check for balanced braces
        brace_count = 0
        for char in query:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            if brace_count < 0:
                raise ValueError("Unbalanced braces in query")
        if brace_count != 0:
            raise ValueError("Unbalanced braces in query")

    async def get_nodes_metrics_and_events(self, timeout: float = 10.0) -> Dict:
        """
        Get node metrics matching the dashboard UI requirements.
        Returns metrics for node ready status, pods, CPU, memory, network I/O, and events.
        """
        try:
            # Get list of nodes first
            nodes_query = self._add_cluster_filter('kube_node_info')
            nodes_result = await self.query(nodes_query)
            
            response = {"nodes": []}
            
            # Extract node names from result
            for result in nodes_result.get('data', {}).get('result', []):
                node_name = result.get('metric', {}).get('node')
                internal_ip = result.get('metric', {}).get('internal_ip')
                if not node_name or not internal_ip:
                    continue
                
                # Define metrics queries for this node
                queries = {
                    'ready_status': self._add_cluster_filter(
                        f'kube_node_status_condition{{node="{node_name}",condition="Ready",status="true"}}'
                    ),
                    'pods_allocated': self._add_cluster_filter(
                        f'kubelet_running_pods{{node="{node_name}"}}'
                    ),
                    'pods_capacity': self._add_cluster_filter(
                        f'kube_node_status_allocatable{{node="{node_name}",resource="pods"}}'
                    ),
                    'cpu_used': self._add_cluster_filter(
                        f'sum(rate(container_cpu_usage_seconds_total{{node="{node_name}"}}[5m])) or vector(0)'
                    ),
                    'cpu_allocatable': self._add_cluster_filter(
                        f'kube_node_status_allocatable{{node="{node_name}",resource="cpu"}} or vector(0)'
                    ),
                    'memory_used': self._add_cluster_filter(
                        f'sum(kube_pod_container_resource_requests{{node="{node_name}",resource="memory"}}) or vector(0)'
                    ),
                    'memory_allocatable': self._add_cluster_filter(
                        f'kube_node_status_allocatable{{node="{node_name}",resource="memory"}} or vector(0)'
                    ),
                    'network_receive': self._add_cluster_filter(
                        f'sum(rate(node_network_receive_bytes_total{{instance="{internal_ip}:9100",device!="lo"}}[5m]))'
                    ),
                    'network_transmit': self._add_cluster_filter(
                        f'sum(rate(node_network_transmit_bytes_total{{instance="{internal_ip}:9100",device!="lo"}}[5m]))'
                    ),
                    'network_io_history': self._add_cluster_filter(
                        f'sum(rate(node_network_receive_bytes_total{{instance="{internal_ip}:9100",device!="lo"}}[5m]) + rate(node_network_transmit_bytes_total{{instance="{internal_ip}:9100",device!="lo"}}[5m]))'
                    ),
                    'events': self._add_cluster_filter(
                        f'sum(kube_event_count{{node="{node_name}"}}) or vector(0)'
                    ),
                    'node_version': self._add_cluster_filter(
                        f'kube_node_info{{node="{node_name}"}}'
                    )
                }

                # Execute all queries in parallel
                results = {}
                tasks = []
                
                # Add instant queries
                for metric_name, query in queries.items():
                    if metric_name != 'network_io_history':  # Skip the network_io_history query
                        tasks.append(self.query(query, timeout=timeout))
                
                # Add range query for network I/O history
                tasks.append(self.query_range(
                    queries['network_io_history'],
                    start="1h",  # Use Prometheus time notation
                    end="now",
                    step="10m",
                    timeout=timeout
                ))
                
                query_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process instant query results
                for metric_name, result in zip([k for k in queries.keys() if k != 'network_io_history'], query_results[:-1]):
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching {metric_name}: {str(result)}")
                        results[metric_name] = 0
                    else:
                        try:
                            # Extract the value from the result
                            if metric_name == 'node_version':
                                results[metric_name] = result['data']['result'][0]['metric'].get('kernel_version', '')
                            else:
                                logger.debug(f"Processing {metric_name} result: {json.dumps(result, indent=2)}")
                                if len(result.get('data', {}).get('result', [])) > 0:
                                    value = float(result['data']['result'][0]['value'][1])
                                    results[metric_name] = value
                                    logger.debug(f"Node {node_name} {metric_name}: {value}")
                                else:
                                    logger.warning(f"No data found for {metric_name}")
                                    results[metric_name] = 0
                                
                                if metric_name in ['cpu_used', 'cpu_allocatable', 'memory_used', 'memory_allocatable']:
                                    if metric_name.startswith('memory'):
                                        logger.debug(f"Node {node_name} {metric_name} in GiB: {value / (1024 * 1024 * 1024)}")
                        except (KeyError, IndexError, ValueError) as e:
                            logger.error(f"Error processing {metric_name} for node {node_name}: {str(e)}")
                            results[metric_name] = 0

                # Process network I/O history result
                network_history_result = query_results[-1]
                network_history = []
                if not isinstance(network_history_result, Exception):
                    try:
                        logger.debug(f"Raw network history result: {json.dumps(network_history_result, indent=2)}")
                        if len(network_history_result.get('data', {}).get('result', [])) > 0:
                            values = network_history_result['data']['result'][0]['values']
                            logger.debug(f"Found {len(values)} data points")
                            for data_point in values:
                                timestamp, value = data_point
                                network_history.append({
                                    "timestamp": timestamp,
                                    "value": round(float(value) / 1024, 2)  # Convert to KiB/s
                                })
                            logger.debug(f"Processed network history: {json.dumps(network_history, indent=2)}")
                        else:
                            logger.warning("No data points found in network history result")
                    except Exception as e:
                        logger.error(f"Error processing network history: {str(e)}", exc_info=True)
                else:
                    logger.error(f"Network history query failed: {str(network_history_result)}")

                # Calculate total network I/O
                network_receive = results.get('network_receive', 0)
                network_transmit = results.get('network_transmit', 0)
                total_network_io = network_receive + network_transmit
                
                logger.debug(f"Network metrics for node {node_name}:")
                logger.debug(f"  Receive: {network_receive} bytes/s")
                logger.debug(f"  Transmit: {network_transmit} bytes/s")
                logger.debug(f"  Total: {total_network_io} bytes/s")

                # Calculate memory used from total and available
                results['memory_used'] = results.get('memory_used', 0)

                # Format the response according to the UI requirements
                node_data = {
                    "name": node_name,
                    "metrics": {
                        "nodeReadyStatus": {
                            "status": "Ready" if results['ready_status'] == 1 else "NotReady",
                            "percentage": 100 if results['ready_status'] == 1 else 0
                        },
                        "podsAvailable": {
                            "current": int(results['pods_allocated']),
                            "desired": int(results['pods_capacity']),
                            "percentage": round((results['pods_allocated'] / results['pods_capacity'] * 100) 
                                            if results['pods_capacity'] > 0 else 0)
                        },
                        "cpuRequests": {
                            "current": round(results['cpu_used'], 2),
                            "allocatable": int(results['cpu_allocatable']),
                            "percentage": round((results['cpu_used'] / results['cpu_allocatable'] * 100)
                                            if results['cpu_allocatable'] > 0 else 0, 2)
                        },
                        "memoryRequests": {
                            "current": int(results['memory_used'] / (1024 * 1024 * 1024)),
                            "allocatable": int(results['memory_allocatable'] / (1024 * 1024 * 1024)),
                            "percentage": round((results['memory_used'] / results['memory_allocatable'] * 100)
                                            if results['memory_allocatable'] > 0 else 0),
                            "unit": "GiB"
                        },
                        "networkIO": {
                            "current": round(total_network_io / 1024, 2),  # Convert to KiB/s
                            "receive": round(network_receive / 1024, 2),  # Convert to KiB/s
                            "transmit": round(network_transmit / 1024, 2),  # Convert to KiB/s
                            "unit": "KiB/s",
                            "trend": "fluctuating",
                            "history": network_history
                        },
                        "events": {
                            "count": "99+" if results['events'] > 99 else str(int(results['events'])),
                            "status": "warning" if results['events'] > 0 else "normal"
                        }
                    },
                
                }
                
                response["nodes"].append(node_data)
            
            return response
            
        except Exception as e:
            logger.error(f"Error fetching node metrics and events: {str(e)}")
            raise
    
    async def query(
        self, 
        query: str, 
        time: Optional[datetime] = None,
        timeout: float = 10.0
    ) -> Dict:
        """Execute an instant query with caching and error handling."""
        cache_key = self._build_cache_key(query, time=time.isoformat() if time else None)
        
        # Check cache first
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        try:
            # Validate query first
            self._validate_query(query)
            
            # Add cluster filter and log the query transformation
            filtered_query = self._add_cluster_filter(query)
            logger.debug(f"Original query: {query}")
            logger.debug(f"Filtered query: {filtered_query}")
            
            params = {
                'query': filtered_query,
                'time': str(int(time.timestamp())) if time else str(int(datetime.now().timestamp()))
            }
            
            logger.debug(f"Query parameters: {params}")
            
            session = await self._get_session()
            async with session.get(
                f"{self.api_url}/api/v1/query",
                params=params,
                timeout=timeout
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                # Cache the result
                self._query_cache[cache_key] = result
                return result

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error executing query: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    async def query_range(
        self,
        query: str,
        start: Union[str, datetime],
        end: Union[str, datetime] = "now",
        step: str = "1m",
        timeout: float = 30.0
    ) -> Dict:
        """Execute a range query with caching and error handling."""
        try:
            # Convert time inputs to timestamps
            start_ts = self._parse_time_input(start)
            end_ts = self._parse_time_input(end)
            
            # Build cache key
            cache_key = self._build_cache_key(
                query, 
                start=start_ts,
                end=end_ts,
                step=step
            )
            
            # Check cache first
            if cache_key in self._range_cache:
                return self._range_cache[cache_key]

            filtered_query = self._add_cluster_filter(query)
            params = {
                'query': filtered_query,
                'start': start_ts,
                'end': end_ts,
                'step': step
            }
            
            logger.debug(f"Range query parameters: {params}")
            
            session = await self._get_session()
            async with session.get(
                f"{self.api_url}/api/v1/query_range",
                params=params,
                timeout=timeout
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                # Cache the result
                self._range_cache[cache_key] = result
                return result

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error executing range query: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error executing range query: {str(e)}")
            raise

    async def get_event_metrics(
        self, 
        time_range: str = "1h",
        timeout: float = 10.0
    ) -> Dict:
        """Get event-related metrics with parallel execution."""
        queries = {
            'total_events': f'sum(kube_event_count{{{self.cluster_label}}})',
            'events_by_type': f'sum by (type) (kube_event_count{{{self.cluster_label}}})',
            'warning_events': f'sum by (reason) (kube_event_count{{type="Warning",{self.cluster_label}}})',
            'events_by_object': f'sum by (involved_object_kind) (kube_event_count{{{self.cluster_label}}})'
        }
        
        try:
            # Execute queries in parallel
            tasks = [
                self.query(query, timeout=timeout) 
                for query in queries.values()
            ]
            results = await asyncio.gather(*tasks)
            
            return dict(zip(queries.keys(), results))
            
        except Exception as e:
            logger.error(f"Error fetching event metrics: {str(e)}")
            raise

    async def get_node_metrics(
        self, 
        node_name: Optional[str] = None,
        timeout: float = 10.0
    ) -> Dict:
        """Get node-specific metrics with parallel execution."""
        node_filter = f'node="{node_name}",' if node_name else ''
        
        queries = {
            'cpu_usage': f'sum(rate(node_cpu_seconds_total{{mode!="idle",{node_filter}{self.cluster_label}}}[5m])) by (instance)',
            'memory_usage': f'node_memory_MemTotal_bytes{{{node_filter}{self.cluster_label}}} - node_memory_MemAvailable_bytes',
            'disk_usage': f'node_filesystem_size_bytes{{{node_filter}{self.cluster_label},mountpoint="/"}} - node_filesystem_free_bytes',
            'network_receive': f'rate(node_network_receive_bytes_total{{{node_filter}{self.cluster_label}}}[5m])',
            'network_transmit': f'rate(node_network_transmit_bytes_total{{{node_filter}{self.cluster_label}}}[5m])'
        }
        
        try:
            # Execute queries in parallel
            tasks = [
                self.query(query, timeout=timeout) 
                for query in queries.values()
            ]
            results = await asyncio.gather(*tasks)
            
            return dict(zip(queries.keys(), results))
            
        except Exception as e:
            logger.error(f"Error fetching node metrics: {str(e)}")
            raise

    async def get_pod_metrics(
        self, 
        namespace: Optional[str] = None,
        timeout: float = 10.0
    ) -> Dict:
        """Get pod-related metrics with parallel execution."""
        namespace_filter = f'namespace="{namespace}",' if namespace else ''
        
        queries = {
            'pod_count': f'count(kube_pod_info{{{namespace_filter}{self.cluster_label}}}) by (namespace)',
            'pod_status': f'sum by (phase) (kube_pod_status_phase{{{namespace_filter}{self.cluster_label}}})',
            'container_restarts': f'sum(kube_pod_container_status_restarts_total{{{namespace_filter}{self.cluster_label}}}) by (pod)',
            'pod_cpu_usage': f'sum(rate(container_cpu_usage_seconds_total{{{namespace_filter}{self.cluster_label}}}[5m])) by (pod)',
            'pod_memory_usage': f'sum(container_memory_working_set_bytes{{{namespace_filter}{self.cluster_label}}}) by (pod)'
        }
        
        try:
            # Execute queries in parallel
            tasks = [
                self.query(query, timeout=timeout) 
                for query in queries.values()
            ]
            results = await asyncio.gather(*tasks)
            
            return dict(zip(queries.keys(), results))
            
        except Exception as e:
            logger.error(f"Error fetching pod metrics: {str(e)}")
            raise

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        await self._close_session()