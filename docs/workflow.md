# Workflow api steps

## Index
[Add Model](#add-model)

[Deploy Model](#deploy-model)

[Add Cluster](#add-cluster)

## Add Model

### Add cloud model workflow

#### Add cloud model to model zoo from seeded model

```json
// Select provider type
{
  "workflow_total_steps": 6,
  "step_number": 1,
  "provider_type": "cloud_model"
}

// Select provider
{
  "workflow_id": "bba793c4-0827-41d1-b463-61ef3ab8800d",
  "step_number": 2,
  "provider_id": "081ffafb-3e72-4c97-ab24-de4e722940b7"
}

// Select cloud model
{
  "workflow_id": "bba793c4-0827-41d1-b463-61ef3ab8800d",
  "step_number": 3,
  "cloud_model_id": "eed59dff-cbed-445d-85e0-298ef0b53591"
}

// Add model details
{
  "workflow_id": "bba793c4-0827-41d1-b463-61ef3ab8800d",
  "step_number": 4,
  "name": "abc",
  "tags": [{"name": "Tag1", "color": "#000000"}, {"name": "Tag2", "color": "#000000"}],
  "modality": "llm",
  "trigger_workflow": true
}
```

#### Add cloud model to model zoo by manual entering

```json
// Select provider type
{
  "workflow_total_steps": 6,
  "step_number": 1,
  "provider_type": "cloud_model"
}

// Select provider
{
  "workflow_id": "27ddc5c2-fbc1-40e9-bfc8-c46f9d75542f",
  "step_number": 2,
  "provider_id": "41e18165-3bbc-45ec-bf2b-81066722e269"
}

// Skip cloud model selection
{
  "workflow_id": "27ddc5c2-fbc1-40e9-bfc8-c46f9d75542f",
  "step_number": 3,
  "cloud_model_id": null
}

// Add model details
{
  "workflow_id": "27ddc5c2-fbc1-40e9-bfc8-c46f9d75542f",
  "step_number": 4,
  "name": "abc",
  "tags": [{"name": "Tag1", "color": "#000000"}, {"name": "Tag2", "color": "#000000"}],
  "modality": "llm",
  "uri": "openai/gpt4",
  "trigger_workflow": true
}
```

### Add local model workflow

```json
// Select provider type
{
    "workflow_total_steps": 6,
    "step_number": 1,
    "trigger_workflow": false,
    "modality": "llm"
}

{
    "workflow_id": "fdf7972c-c649-45dd-9d8e-47b467f04fcb",
    "step_number": 2,
    "trigger_workflow": false,
    "provider_type": "hugging_face"
}
// Add model details
{
    "workflow_id": "fdf7972c-c649-45dd-9d8e-47b467f04fcb",
    "step_number": 3,
    "trigger_workflow": false,
    "name": "Microsoft phi local",
    "uri": "microsoft/Phi-3.5-mini-instruct",
    "author": "microsoft",
    "tags": [
        {
        "name": "text-generation",
        "color": "#66d1DF"
        }
    ],
    "icon": null // icon need to provide for model from disk or url
}
// Select proprietary credential if required
{
    "workflow_id": "4d15bd20-249a-4755-94e8-b8803094133a",
    "step_number": 4,
    "trigger_workflow": true,
    "proprietary_credential_id": "915b5233-85d1-44a4-a694-6d3ecd36b8c6" // not required for public, url, disk models
}
```

## Deploy Model

### Deploy cloud model workflow

```json
// First Screen
{
  "workflow_total_steps": 7,
  "step_number": 1,
  "trigger_workflow": false,
  "project_id": "ffeb0cbe-8caa-461a-aacc-9b7422f44131",
  "model_id": "318dc142-cf44-407d-9b9d-baa47fe4a456"
}
// Second Screen
{
  "workflow_id": "0d9bd0d9-8f11-4a01-bda2-7bfab4735842",
  "step_number": 2,
  "trigger_workflow": false,
  "credential_id": "891f8697-8053-45cf-887f-f2d9cd8cb1ee"
}
// Third Screen
{
  "workflow_id": "0d9bd0d9-8f11-4a01-bda2-7bfab4735842",
  "step_number": 3,
  "trigger_workflow": false,
  "template_id": "891f8697-8053-45cf-887f-f2d9cd8cb1ed"
}
// Fourth Screen
{
  "workflow_id": "0d9bd0d9-8f11-4a01-bda2-7bfab4735842",
  "step_number": 4,
  "trigger_workflow": false,
  "endpoint_name": "test endpoint",
  "deploy_config": {
    "concurrent_requests": 10,
    "avg_sequence_length": 100,
    "avg_context_length": 102
  }
}
// Final Screen
{
  "workflow_id": "0d9bd0d9-8f11-4a01-bda2-7bfab4735842",
  "step_number": 6,
  "trigger_workflow": true,
  "cluster_id": "538072a1-634b-4f8f-bd3b-d2fe6bfe7a50"
}
```

### Deploy local model workflow

```json
// First Screen
{
  "workflow_total_steps": 6,
  "step_number": 1,
  "trigger_workflow": false,
  "project_id": "ffeb0cbe-8caa-461a-aacc-9b7422f44131",
  "model_id": "318dc142-cf44-407d-9b9d-baa47fe4a456"
}
// Second Screen
{
  "workflow_id": "0d9bd0d9-8f11-4a01-bda2-7bfab4735842",
  "step_number": 2,
  "trigger_workflow": false,
  "template_id": "891f8697-8053-45cf-887f-f2d9cd8cb1ed"
}
// Third Screen
{
  "workflow_id": "0d9bd0d9-8f11-4a01-bda2-7bfab4735842",
  "step_number": 3,
  "trigger_workflow": false,
  "endpoint_name": "test endpoint",
  "deploy_config": {
    "concurrent_requests": 10,
    "avg_sequence_length": 100,
    "avg_context_length": 102,
    "per_session_tokens_per_sec": [
      100,
      100
    ],
    "ttft": [
      50,
      1000
    ],
    "e2e_latency": [
      100,
      100
    ]
  }
}
// Final Screen
{
  "workflow_id": "0d9bd0d9-8f11-4a01-bda2-7bfab4735842",
  "step_number": 5,
  "trigger_workflow": true,
  "cluster_id": "538072a1-634b-4f8f-bd3b-d2fe6bfe7a50"
}
```

## Add Cluster
### Add cluster workflow using form data
```curl
curl --location 'https://<base_url>/clusters/clusters' \
--header 'accept: application/json' \
--header 'Authorization: Bearer <token>' \
--form 'step_number="1"' \
--form 'name="Xeon Dev"' \
--form 'icon="icons/providers/openai.png"' \
--form 'ingress_url="https://20.244.107.114:10001"' \
--form 'configuration_file=@"<path_to_yaml>"' \
--form 'workflow_total_steps="3"' \
--form 'trigger_workflow="true"'
```

## Model Security Scan

### Perform model security scan

```json
{
  "workflow_total_steps": 3,
  "step_number": 1,
  "trigger_workflow": true,
  "model_id": "1cda88f7-0747-4f97-8fc3-fca9091f44d5"
}
```

## Add Worker to Endpoint

```json
{
  "workflow_total_steps": 5,
  "step_number": 1,
  "trigger_workflow": false,
  "endpoint_id": "7835329b-97aa-4658-b421-56473aef0500",
  "additional_concurrency": 10
}

{
  "workflow_id": "da939c36-3d5f-434f-9b61-19095024e4e7",
  "step_number": 3,
  "trigger_workflow": true
}
```