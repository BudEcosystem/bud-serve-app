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
  "workflow_id": "f65a1394-df93-46ab-9e08-899f8b49b14c",
  "step_number": 2,
  "provider_id": "1c64b7bf-7302-45f1-bd3d-2405a23a9111"
}

// Select cloud model
{
  "workflow_id": "f65a1394-df93-46ab-9e08-899f8b49b14c",
  "step_number": 3,
  "cloud_model_id": null
}

// Add model details
{
  "workflow_id": "f65a1394-df93-46ab-9e08-899f8b49b14c",
  "step_number": 4,
  "name": "abc",
  "uri": "openai/gpt7",
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

## Deploy Model

### Deploy model workflow

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