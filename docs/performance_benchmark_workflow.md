
# Performance Benchmark Workflow

## Steps for Running Performance Benchmark

### Step 1: Provide Basic Info

Give:
- Name
- Tags
- Description
- Concurrent requests
- Evaluation method: `configuration` or `dataset`

### Step 2: Evaluation Details

- If **evaluation method is "configuration"**: specify `max_input_tokens` and `max_output_tokens`
- If **evaluation method is "dataset"**: choose datasets from the list

### Step 3: Choose Cluster

Select the cluster where the benchmark should run.

### Step 4: Choose Nodes

Pick the specific nodes within the cluster.

### Step 5: Choose Model

Choose the model to run the benchmark.

#### Step 5.1: If Model is Cloud-Based

Select the cloud model credential.

### Step 6: Confirm Benchmark Details

User confirmation of all benchmark details.

### Step 7: Execution Option

Choose either:
- `Simulate` (currently not operational)
- `Run on cluster`

### Step 8: Deployment

Trigger deployment steps. If successful, workflow ends with a success step; otherwise, an error will be shown during deployment.

---

## Example JSON Payloads

### Step 1

<details>
<summary>Configuration method</summary>

```json
{
  "name": "test-111",
  "tags": [],
  "description": "test-111",
  "concurrent_requests": 50,
  "eval_with": "configuration",
  "workflow_total_steps": 9,
  "step_number": 1,
  "trigger_workflow": false
}
```
</details>

<details>
<summary>Dataset method</summary>

```json
{
  "name": "test-111",
  "tags": [],
  "description": "test-111",
  "concurrent_requests": 50,
  "eval_with": "dataset",
  "workflow_total_steps": 9,
  "step_number": 1,
  "trigger_workflow": false
}
```
</details>

---

### Step 2

<details>
<summary>Using Configuration</summary>

```json
{
  "name": "test-111",
  "tags": [],
  "description": "test-111",
  "concurrent_requests": 50,
  "eval_with": "configuration",
  "workflow_id": "dcfddf9b-5f07-47e9-8483-24f546bb7667",
  "step_number": 2,
  "trigger_workflow": false,
  "max_input_tokens": 50,
  "max_output_tokens": 100
}
```
</details>

<details>
<summary>Using Dataset</summary>

```json
{
  "name": "test-111",
  "tags": [],
  "description": "test-111",
  "concurrent_requests": 50,
  "eval_with": "dataset",
  "workflow_id": "dcfddf9b-5f07-47e9-8483-24f546bb7667",
  "step_number": 2,
  "trigger_workflow": false,
  "datset_ids": ["effddf9b-5f07-47e9-8483-24f546bb7667"]
}
```
</details>

---

### Step 3: Choose Cluster

```json
{
  "name": "test-111",
  "tags": [],
  "description": "test-111",
  "concurrent_requests": 50,
  "eval_with": "configuration",
  "workflow_id": "9e4aa5c6-d2df-42ab-8972-a954bd7f6485",
  "step_number": 3,
  "trigger_workflow": false,
  "cluster_id": "1fbb9c3c-4198-4e7e-8a1e-02b86041ea82"
}
```

---

### Step 4: Choose Nodes

Node json is response of GET node-metrics API

```json
{
  "name": "test-111",
  "tags": [],
  "description": "test-111",
  "concurrent_requests": 50,
  "eval_with": "configuration",
  "workflow_id": "9e4aa5c6-d2df-42ab-8972-a954bd7f6485",
  "step_number": 4,
  "trigger_workflow": false,
  "nodes": [
    {
      "hostname": "fl4u42",
      "status": "Ready",
      "devices": [
        {
          "name": "Intel(R) Xeon(R) Platinum 8480+",
          "type": "cpu",
          "device_info": {
            "num_virtual_cores": 224
          },
          "device_config": {
            "peak_fp16_TFLOPS": 329.00454400000007
          },
          "available_count": 8
        }
      ]
    }
  ]
}
```

---

### Step 5: Choose Model

```json
{
  "name": "test-111",
  "tags": [],
  "description": "test-111",
  "concurrent_requests": 50,
  "eval_with": "configuration",
  "workflow_id": "9e4aa5c6-d2df-42ab-8972-a954bd7f6485",
  "step_number": 5,
  "model_id": "30cc08ec-465c-4041-898e-a08e3889cee9"
}
```

---

### Step 6: Cloud Credential (If Cloud Model)

```json
{
  "name": "test-111",
  "tags": [],
  "description": "test-111",
  "concurrent_requests": 50,
  "eval_with": "configuration",
  "workflow_id": "9e4aa5c6-d2df-42ab-8972-a954bd7f6485",
  "step_number": 6,
  "credential_id": "22d4bb50-d574-4cb1-8b09-0b00138b2bcc"
}
```

---

### Step 7: User Confirmation

```json
{
  "name": "test-111",
  "tags": [],
  "description": "test-111",
  "concurrent_requests": 50,
  "eval_with": "configuration",
  "workflow_id": "9e4aa5c6-d2df-42ab-8972-a954bd7f6485",
  "step_number": 7,
  "user_confirmation": true
}
```

---

### Step 8: Execute

```json
{
  "name": "test-111",
  "tags": [],
  "description": "test-111",
  "concurrent_requests": 50,
  "eval_with": "configuration",
  "workflow_id": "9e4aa5c6-d2df-42ab-8972-a954bd7f6485",
  "step_number": 8,
  "run_as_simulation": false,
  "trigger_workflow": true
}
```
