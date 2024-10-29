# Workflow api steps

## Index
[Add Model](#add-model)


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
