A FastAPI backend application called "budapp" that provides functionality for managing machine learning models, clusters, deployments and workflows. Here's a high-level overview of the key components:

Key Components:

1. Auth Module (`/auth`):
- Handles user authentication and token management
- Implements login functionality and JWT token generation

2. Cluster Operations (`/cluster_ops`):
- Manages compute clusters for model deployment
- Handles cluster CRUD operations and status monitoring
- Integrates with external cluster management services

3. Model Operations (`/model_ops`):
- Manages ML models (both local and cloud models)
- Supports model registration, onboarding, scanning and deployment
- Integrates with model repositories and registries

4. Endpoint Operations (`/endpoint_ops`):
- Handles model deployment endpoints
- Manages worker nodes and scaling
- Monitors endpoint health and metrics

5. Project Operations (`/project_ops`):
- Manages projects that can contain multiple models and deployments
- Handles project permissions and collaboration

6. Workflow System (`/workflow_ops`):
- Orchestrates complex operations like model deployment and cluster setup
- Tracks multi-step workflow progress
- Handles workflow state management

7. Core Components (`/commons`, `/core`):
- Configuration management
- Database utilities
- Logging
- Exception handling
- Security utilities

Key Features:

1. Model Management:
- Support for both local and cloud ML models
- Model security scanning and verification
- Model versioning and metadata management

2. Deployment:
- Automated model deployment workflows
- Endpoint scaling and monitoring
- Worker node management

3. Multi-tenancy:
- Project-based isolation
- Role-based access control
- User and permission management

4. Monitoring:
- Cluster health monitoring
- Endpoint metrics collection
- Worker node status tracking

The application uses:
- FastAPI for REST APIs
- SQLAlchemy for database operations
- Pydantic for data validation
- DAPR for microservices communication
- Redis for caching
- JWT for authentication

The codebase follows a clean architecture with clear separation of:
- Routes (API endpoints)
- Services (Business logic)
- Models (Database models)
- Schemas (Data validation)
- CRUD operations (Database operations)

It's designed to be scalable and maintainable with proper exception handling, logging, and modular structure.


Based on the code, here are all the available routes organized by their prefixes:

### Auth Routes (/auth)
- POST `/auth/login` - Login a user with email and password

### Clusters Routes (/clusters)
- POST `/clusters/clusters` - Create cluster workflow
- GET `/clusters/clusters` - List all clusters
- PATCH `/clusters/{cluster_id}` - Edit cluster
- GET `/clusters/{cluster_id}` - Get cluster details
- POST `/clusters/{cluster_id}/delete-workflow` - Delete a cluster
- POST `/clusters/cancel-onboarding` - Cancel cluster onboarding
- GET `/clusters/{cluster_id}/endpoints` - List all endpoints in cluster

### Models Routes (/models)
- GET `/models/` - List all models
- GET `/models/providers` - List all model providers
- POST `/models/cloud-model-workflow` - Add cloud model workflow
- POST `/models/local-model-workflow` - Add local model workflow
- PATCH `/models/{model_id}` - Edit model
- GET `/models/cloud-models/recommended-tags` - List recommended tags
- GET `/models/tags` - Search model tags
- GET `/models/tasks` - Search model tasks
- GET `/models/authors` - Search model authors
- GET `/models/{model_id}` - Get model details
- POST `/models/security-scan` - Scan local model
- POST `/models/cancel-deployment` - Cancel model deployment
- DELETE `/models/{model_id}` - Delete model

### Endpoints Routes (/endpoints)
- GET `/endpoints/` - List all endpoints
- POST `/endpoints/{endpoint_id}/delete-workflow` - Delete endpoint
- GET `/endpoints/{endpoint_id}/workers` - Get endpoint workers
- GET `/endpoints/{endpoint_id}/workers/{worker_id}` - Get worker details
- GET `/endpoints/{endpoint_id}/model-cluster-detail` - Get model cluster details
- POST `/endpoints/delete-worker` - Delete worker
- POST `/endpoints/add-worker` - Add worker to endpoint

### Projects Routes (/projects)
- PATCH `/projects/{project_id}` - Edit project
- GET `/projects/{project_id}/clusters` - List clusters in project

### Users Routes (/users)
- GET `/users/me` - Get current user

### Workflows Routes (/workflows)
- GET `/workflows` - List all workflows
- GET `/workflows/{workflow_id}` - Get workflow details
- PATCH `/workflows/{workflow_id}` - Mark workflow as completed
- DELETE `/workflows/{workflow_id}` - Delete workflow

### Metrics Routes (/metrics)
- POST `/metrics/analytics/request-counts` - Get request count analytics
- POST `/metrics/analytics/request-performance` - Get request performance analytics
- GET `/metrics/count` - Get dashboard statistics

### Other Routes
- GET `/` - Get service info/health
- GET `/health` - Health check
- GET `/sync/configurations` - Sync configurations
- GET `/sync/secrets` - Sync secrets
- GET `/register` - Register service
- GET `/icons` - List icons

This covers all the major API routes defined in the application's routers.

i need to an endpoint which takes the cluster_id as parameter and returns the metrics of the cluster

I'll help you create an endpoint to get cluster metrics. We can add this to the existing cluster_routes.py file. Here's how we can implement it:

1. First, let's create a schema for the metrics response in `cluster_ops/schemas.py`:

```python
class ClusterMetrics(BaseModel):
    """Cluster metrics schema."""

    cluster_id: UUID4
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: Optional[float] = None
    hpu_utilization: Optional[float] = None
    active_endpoints: int
    total_workers: int
    active_workers: int
    uptime: str
    last_sync: datetime

class ClusterMetricsResponse(SuccessResponse):
    """Cluster metrics response schema."""

    metrics: ClusterMetrics
```

2. Add a method in `cluster_ops/services.py` to fetch the metrics:

```python
class ClusterService(SessionMixin):
    async def get_cluster_metrics(self, cluster_id: UUID) -> ClusterMetrics:
        """Get cluster metrics."""
        # First verify if cluster exists
        db_cluster = await ClusterDataManager(self.session).retrieve_by_fields(
            ClusterModel,
            {"id": cluster_id},
            exclude_fields={"status": ClusterStatusEnum.DELETED}
        )

        try:
            # Make request to bud_cluster app to get metrics
            metrics_endpoint = (
                f"{app_settings.dapr_base_url}/v1.0/invoke/"
                f"{app_settings.bud_cluster_app_id}/method/cluster/metrics"
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    metrics_endpoint,
                    params={"cluster_id": str(db_cluster.cluster_id)}
                ) as response:
                    if response.status != 200:
                        raise ClientException("Failed to get cluster metrics")

                    metrics_data = await response.json()

            # Get number of active endpoints
            active_endpoints = await EndpointDataManager(self.session).get_count_by_fields(
                EndpointModel,
                fields={"cluster_id": cluster_id},
                exclude_fields={"status": EndpointStatusEnum.DELETED}
            )

            return ClusterMetrics(
                cluster_id=cluster_id,
                cpu_utilization=metrics_data.get("cpu_utilization", 0.0),
                memory_utilization=metrics_data.get("memory_utilization", 0.0),
                gpu_utilization=metrics_data.get("gpu_utilization"),
                hpu_utilization=metrics_data.get("hpu_utilization"),
                active_endpoints=active_endpoints,
                total_workers=metrics_data.get("total_workers", 0),
                active_workers=metrics_data.get("active_workers", 0),
                uptime=metrics_data.get("uptime", "0s"),
                last_sync=db_cluster.status_sync_at
            )

        except Exception as e:
            logger.exception(f"Failed to get cluster metrics: {e}")
            raise ClientException(
                "Failed to get cluster metrics",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from e
```

3. Add the new endpoint in `cluster_ops/cluster_routes.py`:

```python
@cluster_router.get(
    "/{cluster_id}/metrics",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
        status.HTTP_200_OK: {
            "model": ClusterMetricsResponse,
            "description": "Successfully retrieved cluster metrics",
        },
    },
    description="Get metrics for a specific cluster"
)
async def get_cluster_metrics(
    cluster_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
) -> Union[ClusterMetricsResponse, ErrorResponse]:
    """Get metrics for a specific cluster."""
    try:
        metrics = await ClusterService(session).get_cluster_metrics(cluster_id)
        return ClusterMetricsResponse(
            metrics=metrics,
            message="Successfully retrieved cluster metrics",
            code=status.HTTP_200_OK,
            object="cluster.metrics"
        ).to_http_response()
    except ClientException as e:
        logger.exception(f"Failed to get cluster metrics: {e}")
        return ErrorResponse(
            code=e.status_code,
            message=e.message
        ).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get cluster metrics: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Failed to get cluster metrics"
        ).to_http_response()
```

This implementation:

1. Creates a new GET endpoint at `/clusters/{cluster_id}/metrics`
2. Returns cluster metrics including:
   - CPU/Memory/GPU/HPU utilization
   - Number of active endpoints
   - Total and active workers
   - Cluster uptime
   - Last sync time
3. Handles proper error cases
4. Includes authentication via the current_user dependency
5. Makes use of the existing database session management
6. Integrates with the bud_cluster service via Dapr

You can call this endpoint like:
```bash
GET /clusters/123e4567-e89b-12d3-a456-426614174000/metrics
```

Response will look like:
```json
{
    "code": 200,
    "object": "cluster.metrics",
    "message": "Successfully retrieved cluster metrics",
    "metrics": {
        "cluster_id": "123e4567-e89b-12d3-a456-426614174000",
        "cpu_utilization": 45.5,
        "memory_utilization": 60.2,
        "gpu_utilization": 80.0,
        "hpu_utilization": null,
        "active_endpoints": 3,
        "total_workers": 10,
        "active_workers": 8,
        "uptime": "5d 12h 30m",
        "last_sync": "2024-01-20T15:30:00.000Z"
    }
}
```


I'll help you update the endpoint to use Prometheus metrics. Here's the revised implementation:

1. First, update the schema in `cluster_ops/schemas.py`:

```python
class ClusterNodeNetwork(BaseModel):
    """Network metrics for a cluster node."""
    total_receive_mbps: float
    total_transmit_mbps: float
    total_bandwidth_mbps: float
    total_errors: float

class ClusterNodeMetrics(BaseModel):
    """Metrics for a single node in the cluster."""
    memory: Dict[str, float]  # total_gib, used_gib, available_gib, usage_percent
    cpu: Dict[str, float]     # cpu_usage_percent
    disk: Dict[str, Dict]     # paths with disk metrics
    gpu: Dict[str, float]     # memory and utilization metrics
    hpu: Dict[str, float]     # memory and utilization metrics
    network: Dict[str, Union[Dict[str, Dict], Dict[str, float]]]  # interfaces and summary

class ClusterSummaryMetrics(BaseModel):
    """Summary metrics for the entire cluster."""
    total_nodes: int
    memory: Dict[str, float]  # total_gib, used_gib, available_gib, usage_percent
    disk: Dict[str, float]    # total_gib, used_gib, available_gib, usage_percent
    gpu: Dict[str, float]     # memory and utilization metrics
    hpu: Dict[str, float]     # memory and utilization metrics
    cpu: Dict[str, float]     # average_usage_percent
    network: Dict[str, float] # network metrics

class ClusterMetricsResponse(SuccessResponse):
    """Cluster metrics response schema."""
    nodes: Dict[str, ClusterNodeMetrics]
    cluster_summary: ClusterSummaryMetrics
```

2. Add the ClusterMetricsFetcher class in `cluster_ops/utils.py`:

```python
# cluster_ops/utils.py

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
        # Add all the query logic from your provided code here
        # This is the same implementation as in your example
        queries = {
            'clusters': 'count(node_uname_info) by (cluster)',
            'memory_total': 'node_memory_MemTotal_bytes / 1024 / 1024 / 1024',
            # ... rest of your queries ...
        }

        # Rest of your implementation
        # ...
```

3. Update the service method in `cluster_ops/services.py`:

```python
from budapp.commons.config import app_settings
from .utils import ClusterMetricsFetcher

class ClusterService(SessionMixin):
    async def get_cluster_metrics(self, cluster_id: UUID) -> Dict[str, Any]:
        """Get cluster metrics from Prometheus."""
        # First verify if cluster exists
        db_cluster = await ClusterDataManager(self.session).retrieve_by_fields(
            ClusterModel,
            {"id": cluster_id},
            exclude_fields={"status": ClusterStatusEnum.DELETED}
        )

        try:
            # Initialize metrics fetcher
            prometheus_url = app_settings.prometheus_url  # Add this to your config
            metrics_fetcher = ClusterMetricsFetcher(prometheus_url)

            # Get metrics for all clusters
            all_metrics = metrics_fetcher.get_cluster_metrics()

            if not all_metrics:
                raise ClientException("Failed to fetch metrics from Prometheus")

            # Get metrics for specific cluster
            cluster_name = db_cluster.name  # assuming cluster name is used in Prometheus
            if cluster_name not in all_metrics:
                raise ClientException(f"No metrics found for cluster: {cluster_name}")

            cluster_metrics = all_metrics[cluster_name]

            return {
                "nodes": cluster_metrics["nodes"],
                "cluster_summary": cluster_metrics["cluster_summary"]
            }

        except Exception as e:
            logger.exception(f"Failed to get cluster metrics: {e}")
            raise ClientException(
                "Failed to get cluster metrics",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from e
```

4. The route in `cluster_ops/cluster_routes.py` remains similar but uses the new schemas:

```python
@cluster_router.get(
    "/{cluster_id}/metrics",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
        status.HTTP_200_OK: {
            "model": ClusterMetricsResponse,
            "description": "Successfully retrieved cluster metrics",
        },
    },
    description="Get detailed metrics for a specific cluster including CPU, memory, disk, GPU, HPU, and network statistics"
)
async def get_cluster_metrics(
    cluster_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
) -> Union[ClusterMetricsResponse, ErrorResponse]:
    """Get detailed metrics for a specific cluster."""
    try:
        metrics = await ClusterService(session).get_cluster_metrics(cluster_id)
        return ClusterMetricsResponse(
            nodes=metrics["nodes"],
            cluster_summary=metrics["cluster_summary"],
            message="Successfully retrieved cluster metrics",
            code=status.HTTP_200_OK,
            object="cluster.metrics"
        ).to_http_response()
    except ClientException as e:
        logger.exception(f"Failed to get cluster metrics: {e}")
        return ErrorResponse(
            code=e.status_code,
            message=e.message
        ).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get cluster metrics: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Failed to get cluster metrics"
        ).to_http_response()
```

5. Add the Prometheus URL to your config in `commons/config.py`:

```python
class AppConfig(BaseConfig):
    # ... existing config ...
    prometheus_url: str = Field("http://localhost:9090", alias="PROMETHEUS_URL")
```

This implementation:
- Uses Prometheus to fetch detailed metrics
- Provides comprehensive metrics about CPU, memory, disk, GPU, HPU, and network
- Returns both node-level and cluster-summary metrics
- Maintains the existing error handling and authentication
- Uses proper typing and validation
- Follows the existing project structure and patterns

You can call the endpoint:
```bash
GET /clusters/{cluster_id}/metrics
```

And it will return detailed metrics in the format specified by the ClusterMetricsResponse schema.
