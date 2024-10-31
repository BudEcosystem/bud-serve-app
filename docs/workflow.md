# Workflow api steps

## Index
[Add Model](#add-model)

[Deploy Model](#deploy-model)

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
  "workflow_id": "ba3fd9d3-ce02-4dd7-ae21-4a79ea6ca034",
  "step_number": 2,
  "provider_id": "41e18165-3bbc-45ec-bf2b-81066722e269"
}

// Select cloud model
{
  "workflow_id": "ba3fd9d3-ce02-4dd7-ae21-4a79ea6ca034",
  "step_number": 3,
  "cloud_model_id": "666316bd-a836-4eb6-a200-a20f46e45db4"
}

// Add model details
{
  "workflow_id": "ba3fd9d3-ce02-4dd7-ae21-4a79ea6ca034",
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
  "workflow_id": "95a98482-4e00-48d6-8514-137914cd786f",
  "step_number": 2,
  "trigger_workflow": false,
  "template_id": "791e2219-5377-4aba-b0f5-a61b083d6bec"
}
// Third Screen
{
  "workflow_id": "95a98482-4e00-48d6-8514-137914cd786f",
  "step_number": 3,
  "trigger_workflow": false,
  "endpoint_name": "test endpoint",
  "deploy_config": {
    "concurrent_requests": 10,
    "avg_sequence_length": 2,
    "avg_context_length": 3,
    "per_session_tokens_per_sec": [
      0,
      5
    ],
    "ttft": [
      0,
      6
    ]
  }
}
// Final Screen
{
  "workflow_id": "95a98482-4e00-48d6-8514-137914cd786f",
  "step_number": 5,
  "trigger_workflow": true, // consider this as final step
  "cluster_id": "57e574e3-8208-4ba1-bb91-28c1d6e02b32"
}
```