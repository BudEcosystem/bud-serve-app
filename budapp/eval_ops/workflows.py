#  -----------------------------------------------------------------------------
#  Copyright (c) 2024 Bud Ecosystem Inc.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  -----------------------------------------------------------------------------

"""This file contains the workflows for evaluation data synchronization."""

import uuid
from datetime import timedelta
from typing import Any, Dict, Optional

import dapr.ext.workflow as wf
from budmicroframe.commons.schemas import WorkflowStep
from budmicroframe.shared.dapr_workflow import DaprWorkflow

from ..commons import logging


logger = logging.get_logger(__name__)

dapr_workflow = DaprWorkflow()

retry_policy = wf.RetryPolicy(
    first_retry_interval=timedelta(seconds=1),
    max_number_of_attempts=3,
    backoff_coefficient=2,
    max_retry_interval=timedelta(seconds=10),
    retry_timeout=timedelta(seconds=100),
)


class EvalDataSyncWorkflows:
    """Workflows for synchronizing evaluation datasets and traits from cloud repository."""

    def __init__(self) -> None:
        """Initialize the EvalDataSyncWorkflows class."""
        self.dapr_workflow = DaprWorkflow()

    @dapr_workflow.register_activity
    @staticmethod
    def check_and_sync_eval_data(ctx: wf.WorkflowActivityContext, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Check cloud repository for updated evaluation data and sync if needed.
        
        This activity will:
        1. Check a cloud URL for the latest version of eval datasets and traits
        2. Compare with current local version
        3. Download and migrate if a newer version is available
        
        Args:
            ctx: Workflow activity context
            kwargs: Dictionary containing sync parameters
            
        Returns:
            Dictionary with sync results
        """
        repository_url = kwargs.get("repository_url", "https://webhook.site/21df9b68-74a3-40a5-ad85-51d6fe7d6b3b/manifest.json")
        force_sync = kwargs.get("force_sync", False)
        
        try:
            logger.info("Checking cloud repository for evaluation data updates: %s", repository_url)
            
            # TODO: Implement actual logic to:
            # 1. Fetch manifest from repository_url
            # 2. Check version against local database
            # 3. Download new data if available
            # 4. Run migration/update process
            
            # For now, just log the dummy check
            result = {
                "checked": True,
                "repository_url": repository_url,
                "current_version": "1.0.0",  # Would fetch from database
                "latest_version": "1.0.0",   # Would fetch from cloud
                "updated": False,
                "message": "No updates available"
            }
            
            logger.info("Evaluation data sync check completed: %s", result)
            return result
            
        except Exception as e:
            logger.exception("Failed to check/sync evaluation data: %s", e)
            return {
                "checked": False,
                "error": str(e),
                "message": "Failed to sync evaluation data"
            }

    @dapr_workflow.register_workflow
    @staticmethod
    def run_eval_data_sync(ctx: wf.DaprWorkflowContext, payload: Dict[str, Any]):
        """Run the evaluation data synchronization workflow.
        
        Args:
            ctx: Dapr workflow context
            payload: Dictionary containing workflow parameters
        """
        logger.info("Evaluation data sync workflow started")
        logger.info("Is workflow replaying: %s", ctx.is_replaying)

        workflow_name = "eval_data_sync_workflow"
        workflow_id = ctx.instance_id

        repository_url = payload.get("repository_url", "https://webhook.site/21df9b68-74a3-40a5-ad85-51d6fe7d6b3b/manifest.json")
        force_sync = payload.get("force_sync", False)

        result = yield ctx.call_activity(
            EvalDataSyncWorkflows.check_and_sync_eval_data,
            input={
                "repository_url": repository_url,
                "force_sync": force_sync
            },
            retry_policy=retry_policy,
        )
        
        logger.info("Evaluation data sync workflow completed with result: %s", result)
        logger.info("Workflow %s with id %s completed", workflow_name, workflow_id)
        
        return result

    def __call__(self, repository_url: Optional[str] = None, force_sync: bool = False, workflow_id: Optional[str] = None):
        """Trigger the evaluation data sync workflow.
        
        Args:
            repository_url: URL of the cloud repository to check
            force_sync: Force synchronization even if versions match
            workflow_id: Optional workflow ID
            
        Returns:
            Workflow scheduling response
        """
        response = dapr_workflow.schedule_workflow(
            workflow_name="run_eval_data_sync",
            workflow_input={
                "repository_url": repository_url or "https://webhook.site/21df9b68-74a3-40a5-ad85-51d6fe7d6b3b/manifest.json",
                "force_sync": force_sync
            },
            workflow_id=str(workflow_id or uuid.uuid4()),
            workflow_steps=[
                WorkflowStep(
                    id="eval_data_sync",
                    title="Evaluation Data Synchronization",
                    description="Check and sync evaluation datasets and traits from cloud repository",
                )
            ],
        )

        return response