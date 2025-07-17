# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

Bud Serve App is a FastAPI-based microservice for the Bud Runtime ecosystem that manages AI/ML model deployments, clusters, endpoints, and provides comprehensive infrastructure for AI model orchestration. The application uses:
- **Dapr** for microservice communication and workflow orchestration
- **PostgreSQL** for data persistence with Alembic migrations
- **Redis** for caching and session management
- **MinIO** for model/dataset storage
- **Keycloak** for authentication and multi-tenant support
- **Prometheus/Grafana** for metrics and monitoring

### Module Structure
Each module in `budapp/` follows a consistent pattern:
- `models.py` - SQLAlchemy models
- `schemas.py` - Pydantic schemas for API validation
- `crud.py` - Database operations
- `services.py` - Business logic
- `*_routes.py` - FastAPI route definitions

Key modules:
- `auth/` - Authentication with Keycloak integration
- `benchmark_ops/` - Model benchmarking and performance testing
- `cluster_ops/` - Cluster management and workflows (both local and cloud)
- `commons/` - Shared utilities, config, dependencies, and security
- `core/` - Core functionality, notifications, and metadata
- `credential_ops/` - Management of AI service and cloud provider credentials
- `dataset_ops/` - Dataset management for model training/evaluation
- `endpoint_ops/` - Endpoint deployments and management
- `metric_ops/` - Metrics collection and Prometheus integration
- `model_ops/` - Model management (cloud and local)
- `permissions/` - Fine-grained permission management
- `playground_ops/` - Interactive model testing environment
- `project_ops/` - Project management and organization
- `router_ops/` - AI model routing, load balancing, and fallback strategies
- `shared/` - External service integrations (Dapr, MinIO, Redis, Grafana, notifications)
- `user_ops/` - User management functionality
- `workflow_ops/` - Dapr workflow definitions and management

## Development Commands

### Start Development Environment
```bash
# Start all services (PostgreSQL, Redis, Keycloak, MinIO, Dapr)
./deploy/start_dev.sh

# Stop all services
./deploy/stop_dev.sh
```

### Database Operations
```bash
# Apply migrations
alembic -c ./budapp/alembic.ini upgrade head

# Create new migration
alembic -c ./budapp/alembic.ini revision --autogenerate -m "description"
```

### Testing
```bash
# Run all tests with Dapr
pytest --dapr-http-port 3510 --dapr-api-token <TOKEN>

# Run specific test file
pytest tests/test_cluster_metrics.py --dapr-http-port 3510 --dapr-api-token <TOKEN>
```

### Code Quality
```bash
# Format code with Ruff
ruff format .

# Lint code
ruff check .

# Type checking
mypy budapp/
```

## Key Development Patterns

### API Endpoints
All endpoints follow RESTful conventions with consistent response schemas:
- Use `APIResponseSchema` for standard responses
- Include proper status codes and error handling
- Implement pagination for list endpoints using `PaginationQuery`
- Endpoints are organized by module (e.g., `/clusters`, `/models`, `/endpoints`, `/routers`)

### Dapr Workflows
Workflows are defined in `workflows.py` files and registered in `scheduler.py`:
- Each workflow must handle exceptions and update status appropriately
- Use `WorkflowActivityContext` for activity functions
- Store workflow instance IDs in the database for tracking
- Key workflows include:
  - `CloudModelSyncWorkflows`: Syncs cloud-hosted models
  - `ClusterRecommendedSchedulerWorkflows`: Manages cluster recommendations

### Database Models
- All models inherit from `Base` with standard fields (id, created_at, updated_at)
- Use UUID primary keys
- Include proper relationships and indexes
- Handle soft deletes with status fields rather than actual deletion
- Models use timezone-aware timestamps with `func.timezone('UTC', func.now())`

### Authentication & Authorization
- All protected endpoints require JWT tokens via `get_current_user` dependency
- Multi-tenant support through Keycloak realms
- Role-based access control with permissions stored in database
- Project-level permissions for fine-grained access control
- Support for both user credentials and proprietary API credentials

### Error Handling
- Use custom exceptions from `commons/exceptions.py`
- Implement proper error responses with meaningful messages
- Log errors with structured logging via structlog

### External Services Integration
- **MinIO**: Object storage for models and datasets
- **Redis**: Caching layer for performance optimization
- **Prometheus/Grafana**: Metrics collection and monitoring
- **Dapr**: Service-to-service communication and workflows
- All integrations use dedicated service classes in `shared/` directory

## Environment Configuration

Required environment variables (see `.env.sample`):
- Database: `POSTGRES_*`
- Redis: `REDIS_*`
- Keycloak: `KEYCLOAK_*`
- MinIO: `MINIO_*`
- Dapr: `DAPR_*`
- Application: `APP_NAME`, `SECRET_KEY`, `ALGORITHM`

## Testing Guidelines

- Tests require Dapr to be running with proper API token
- Use async test functions with `pytest.mark.asyncio`
- Mock external services (Keycloak, MinIO) in tests
- Test database operations use transactions that rollback after each test
- Include both positive and negative test cases
- Test files are located in the `tests/` directory

## Key Features

### Model Management
- Support for both local and cloud-hosted models
- Model versioning and metadata tracking
- Integration with HuggingFace Hub for model discovery
- Model security scanning and license validation
- Quantization support for optimized deployments

### Cluster Management
- Local and cloud cluster support (AWS, Azure, GCP, etc.)
- Dynamic cluster provisioning and scaling
- Resource monitoring and optimization
- GPU/CPU resource allocation
- Cluster health monitoring and status tracking

### Endpoint Management
- Model deployment to clusters
- Auto-scaling configurations
- Load balancing and routing
- Health checks and monitoring
- Support for multiple model configurations

### Router System
- Multi-model routing with fallback strategies
- Rate limiting (TPM/RPM) per endpoint
- Weighted routing for A/B testing
- Cooldown periods for failed endpoints
- Project-scoped router management

### Benchmarking Framework
- Performance testing for models
- Dataset-based evaluation
- Metrics collection and comparison
- Request/response time tracking
- Resource utilization monitoring

### Playground Environment
- Interactive model testing
- Chat session management
- Custom model configurations
- Message history tracking
- Real-time inference testing

## Database Schema Highlights

- **Multi-tenancy**: All models include `realm_id` for tenant isolation
- **Soft Deletes**: Status fields instead of hard deletes
- **Audit Trail**: `created_at`, `updated_at` timestamps on all models
- **UUID Keys**: All primary keys use UUIDs for distributed systems
- **Relationships**: Proper foreign key constraints and indexes