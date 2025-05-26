# Model Onboarding

This document provides comprehensive guidance on onboarding new models using the Bud Serve App's workflow APIs. Two main POST endpoints are exposed by the backend to facilitate adding a model in incremental steps.

- `POST /models/cloud-model-workflow` &ndash; add a **proprietary ** model from a provider such as OpenAI or Anthropic.
- `POST /models/local-model-workflow` &ndash; add a **local** model from HuggingFace, a direct URL or an existing directory on disk.

Each endpoint expects a small JSON payload describing the current step in the workflow. The frontend will call the endpoint multiple times, progressing from step 1 to the final confirmation step. A `workflow_id` is returned on the first call and must be supplied for all following calls. This allows the backend to store progress and resume if the user revisits the form later.

---

## Cloud Model Workflow

The cloud model workflow is used when adding a model hosted by a cloud provider. The user selects a provider and optionally chooses from seeded models. The workflow is complete once the model is created in the Bud application model repository.

### Steps Overview

1. **Choose Provider Type** &ndash; specify `provider_type="cloud_model"`.
2. **Select Provider** &ndash; choose one of the available providers by `provider_id`.
3. **Select Cloud Model** &ndash; either select a seeded model (`cloud_model_id`) or skip for manual input.
4. **Model Details** &ndash; provide `name`, `tags`, optional `modality` and `uri`, then set `trigger_workflow=true` to finalize.

At each step the workflow service validates the input and stores the step data in the database. When `trigger_workflow` is true, a new model is created. The service compiles the stored steps, inserts the model, marks the workflow completed and returns a workflow object.

### Sequence Diagram

```mermaid
sequenceDiagram
    participant UI
    participant budapp
    participant DB
    UI->>budapp: POST /models/cloud-model-workflow (step N)
    budapp->>DB: create/update Workflow and WorkflowStep
    budapp-->>UI: workflow data
    Note over budapp,DB: When trigger_workflow=true
    budapp->>DB: validate & insert Model
    budapp->>DB: mark Workflow completed
    budapp-->>UI: final workflow data
```

### Flowchart

```mermaid
flowchart TD
    A[Select provider type] --> B[Choose provider]
    B --> C{Seeded model?}
    C -- yes --> D[Pick seeded cloud model]
    C -- no --> E[Enter URI manually]
    D --> F[Add name & tags]
    E --> F[Add name & tags]
    F --> G[trigger_workflow=true]
    G --> H[Model created]
```

### Notifications

The cloud workflow is relatively short, The final model appears immediately in the UI.

## Local Model Workflow

The local model workflow is meant for models that must be downloaded or copied to the Budmodel. This includes HuggingFace models, direct download links and models already stored on the cluster disk. Because downloading and extracting can take time, the workflow stores progress and provides live status updates.

### Steps Overview

1. **Choose Provider Type** &ndash; `hugging_face`, `url` or `disk`.
2. **Model Details** &ndash; provide `name`, `uri` (repo ID, URL or folder path), optional `author`, `tags` and `icon`.
3. **Optional Credential** &ndash; for gated/private HuggingFace models supply `proprietary_credential_id`.
4. **Trigger Extraction** &ndash; set `trigger_workflow=true` to start the extraction process.

Once triggered, `LocalModelWorkflowService` initiates a background workflow in the **budmodel** microservice. The Dapr workflow runs several activities: validating the URI, downloading files, extracting metadata, creating model licenses and finally saving the model to the registry. Events from each activity are streamed back to the frontend so the user can monitor progress.

### Sequence Diagram

```mermaid
sequenceDiagram
    participant UI
    participant budapp
    participant budmodel
    participant DB
    UI->>budapp: POST /models/local-model-workflow (step N)
    budapp->>DB: create/update Workflow and WorkflowStep
    budapp-->>UI: workflow data
    Note over budapp,budmodel: When trigger_workflow=true
    budapp->>budmodel: start extraction workflow
    budmodel->>budapp: progress events
    budapp->>DB: create Model from extracted info
    budapp->>DB: mark Workflow completed
    budapp-->>UI: final workflow data
```

### Flowchart

```mermaid
flowchart TD
    AA[Select provider type] --> BB[Enter model details]
    BB --> CC{Need credential?}
    CC -- yes --> DD[Provide credential]
    CC -- no --> EE[Skip]
    DD --> FF[trigger_workflow=true]
    EE --> FF[trigger_workflow=true]
    FF --> GG[budmodel extraction]
    GG --> HH{success?}
    HH -- yes --> II[Model created]
    HH -- no --> JJ[Workflow failed]
```