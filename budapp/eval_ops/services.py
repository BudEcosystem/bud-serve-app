import uuid
from typing import List, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.eval_ops.models import (
    RunStatusEnum,
    EvaluationStatusEnum,
)
from budapp.eval_ops.models import Experiment as ExperimentModel
from budapp.eval_ops.models import ExpTrait as TraitModel
from budapp.eval_ops.models import ExpDataset as DatasetModel
from budapp.eval_ops.models import ExpTraitsDatasetPivot as PivotModel
from budapp.eval_ops.models import (
    Run as RunModel,
    Evaluation as EvaluationModel,
    ExpMetric as MetricModel,
    ExpRawResult as RawResultModel,
)
from budapp.eval_ops.schemas import (
    CreateExperimentRequest,
    CreateRunRequest,
    UpdateExperimentRequest,
    UpdateRunRequest,
    UpdateEvaluationRequest,
    CreateDatasetRequest,
    UpdateDatasetRequest,
    DatasetFilter,
)
from budapp.eval_ops.schemas import (
    Experiment as ExperimentSchema,
)
from budapp.eval_ops.schemas import (
    Run as RunSchema,
    RunWithEvaluations as RunWithEvaluationsSchema,
)
from budapp.eval_ops.schemas import (
    Evaluation as EvaluationSchema,
    EvaluationWithResults as EvaluationWithResultsSchema,
)
from budapp.eval_ops.schemas import (
    Trait as TraitSchema,
)
from budapp.eval_ops.schemas import (
    ExpDataset as DatasetSchema,
)


logger = logging.get_logger(__name__)

class ExperimentService:
    """Service layer for Experiment operations.

    Methods:
        - create_experiment: create and persist a new Experiment.
        - list_experiments: retrieve all non-deleted Experiments for a user.
        - update_experiment: apply updates to an existing Experiment.
        - delete_experiment: perform a soft delete on an Experiment.
        - create_run: create a run with multiple evaluations.
        - list_runs: list runs for an experiment.
        - get_run_with_evaluations: get a run with its evaluations.
        - update_run: update a run.
        - delete_run: delete a run.
        - list_evaluations: list evaluations within a run.
        - get_evaluation_with_results: get an evaluation with metrics and results.
        - update_evaluation: update an evaluation.
        - list_traits: list Trait entries with optional filters and pagination.
        - get_dataset_by_id: get a dataset by ID with associated traits.
        - list_datasets: list datasets with optional filters and pagination.
        - create_dataset: create a new dataset with traits.
        - update_dataset: update an existing dataset and its traits.
        - delete_dataset: delete a dataset and its trait associations.
    """

    def __init__(self, session: Session):
        """Initialize the service with a database session.

        Parameters:
            session (Session): SQLAlchemy database session.
        """
        self.session = session

    def create_experiment(self, req: CreateExperimentRequest, user_id: uuid.UUID) -> ExperimentSchema:
        """Create a new Experiment record.

        Parameters:
            req (CreateExperimentRequest): Payload containing name, description, project_id.
            user_id (uuid.UUID): ID of the user creating the experiment.

        Returns:
            ExperimentSchema: Pydantic schema of the created Experiment.

        Raises:
            HTTPException(status_code=500): If database insertion fails.
        """
        ev = ExperimentModel(
            name=req.name,
            description=req.description,
            project_id=req.project_id,
            created_by=user_id,
            status="active",
        )
        try:
            self.session.add(ev)
            self.session.commit()
            self.session.refresh(ev)
        except Exception as e:
            self.session.rollback()
            logger.debug(f"Failed to create experiment: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create experiment"
            ) from e
        return ExperimentSchema.from_orm(ev)

    def list_experiments(self, user_id: uuid.UUID) -> List[ExperimentSchema]:
        """List all non-deleted Experiments for a given user.

        Parameters:
            user_id (uuid.UUID): ID of the user whose experiments to list.

        Returns:
            List[ExperimentSchema]: List of Pydantic schemas for each Experiment.

        Raises:
            HTTPException(status_code=500): If database query fails.
        """
        try:
            q = (
                self.session.query(ExperimentModel)
                .filter(
                    ExperimentModel.created_by == user_id,
                    ExperimentModel.status != "deleted",
                )
                .order_by(ExperimentModel.created_at.desc())
            )
            evs = q.all()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list experiments"
            ) from e
        return [ExperimentSchema.from_orm(e) for e in evs]

    def update_experiment(
        self,
        ev_id: uuid.UUID,
        req: UpdateExperimentRequest,
        user_id: uuid.UUID,
    ) -> ExperimentSchema:
        """Update fields of an existing Experiment.

        Parameters:
            ev_id (uuid.UUID): ID of the experiment to update.
            req (UpdateExperimentRequest): Payload with optional name/description.
            user_id (uuid.UUID): ID of the user attempting the update.

        Returns:
            ExperimentSchema: Pydantic schema of the updated Experiment.

        Raises:
            HTTPException(status_code=404): If experiment not found or access denied.
            HTTPException(status_code=500): If database update fails.
        """
        ev = self.session.get(ExperimentModel, ev_id)
        if not ev or ev.created_by != user_id or ev.status == "deleted":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or access denied"
            )
        if req.name is not None:
            ev.name = req.name
        if req.description is not None:
            ev.description = req.description
        try:
            self.session.commit()
            self.session.refresh(ev)
        except Exception as e:
            self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update experiment"
            ) from e
        return ExperimentSchema.from_orm(ev)

    def delete_experiment(self, ev_id: uuid.UUID, user_id: uuid.UUID) -> None:
        """Soft-delete an Experiment by setting its status to 'deleted'.

        Parameters:
            ev_id (uuid.UUID): ID of the experiment to delete.
            user_id (uuid.UUID): ID of the user attempting the delete.

        Raises:
            HTTPException(status_code=404): If experiment not found or access denied.
            HTTPException(status_code=500): If database commit fails.
        """
        ev = self.session.get(ExperimentModel, ev_id)
        if not ev or ev.created_by != user_id or ev.status == "deleted":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or access denied"
            )
        ev.status = "deleted"
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete experiment"
            ) from e

    def list_traits(
        self,
        offset: int = 0,
        limit: int = 10,
        name: Optional[str] = None,
        unique_id: Optional[str] = None,
    ) -> Tuple[List[TraitSchema], int]:
        """List Trait entries with optional filters and pagination.

        Parameters:
            offset (int): Number of records to skip (for pagination).
            limit (int): Maximum number of records to return.
            name (Optional[str]): Optional case-insensitive substring filter on trait name.
            unique_id (Optional[str]): Optional exact UUID filter on trait ID.

        Returns:
            Tuple[List[TraitSchema], int]: A tuple of (list of TraitSchema, total count).

        Raises:
            HTTPException(status_code=500): If database query fails.
        """
        try:
            q = self.session.query(TraitModel)
            
            # Apply filters
            if name:
                q = q.filter(TraitModel.name.ilike(f"%{name}%"))
            if unique_id:
                try:
                    trait_uuid = uuid.UUID(unique_id)
                    q = q.filter(TraitModel.id == trait_uuid)
                except ValueError:
                    # Invalid UUID format, return empty results
                    return [], 0

            # Get total count before applying pagination
            total_count = q.count()
            
            # Apply pagination and get results
            traits = q.offset(offset).limit(limit).all()
            
            # For each trait, get associated datasets
            trait_schemas = []
            for trait in traits:
                # Get datasets associated with this trait
                datasets_query = (
                    self.session.query(DatasetModel)
                    .join(PivotModel, DatasetModel.id == PivotModel.dataset_id)
                    .filter(PivotModel.trait_id == trait.id)
                    .all()
                )
                
                # Convert datasets to DatasetBasic schema
                from budapp.eval_ops.schemas import DatasetBasic
                datasets = [
                    DatasetBasic(
                        id=dataset.id,
                        name=dataset.name,
                        description=dataset.description,
                        estimated_input_tokens=dataset.estimated_input_tokens,
                        estimated_output_tokens=dataset.estimated_output_tokens,
                        modalities=dataset.modalities,
                        sample_questions_answers=dataset.sample_questions_answers,
                        advantages_disadvantages=dataset.advantages_disadvantages,
                    )
                    for dataset in datasets_query
                ]
                
                trait_schema = TraitSchema(
                    id=trait.id,
                    name=trait.name,
                    description=trait.description or "",
                    category="",  # Optional field, can be empty
                    exps_ids=[],  # Optional field for UI, can be empty
                    datasets=datasets,
                )
                trait_schemas.append(trait_schema)
            
            return trait_schemas, total_count

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list traits"
            ) from e

    # ------------------------ Run Methods ------------------------

    def create_run(self, experiment_id: uuid.UUID, req: CreateRunRequest, user_id: uuid.UUID) -> RunWithEvaluationsSchema:
        """Create a new run with multiple evaluations.

        Parameters:
            experiment_id (uuid.UUID): ID of the parent experiment.
            req (CreateRunRequest): Payload containing run details and evaluations.
            user_id (uuid.UUID): ID of the user creating the run.

        Returns:
            RunWithEvaluationsSchema: Pydantic schema of the created run with evaluations.

        Raises:
            HTTPException(status_code=404): If experiment not found or access denied.
            HTTPException(status_code=500): If database insertion fails.
        """
        # Check experiment exists and user has access
        ev = self.session.get(ExperimentModel, experiment_id)
        if not ev or ev.created_by != user_id or ev.status == "deleted":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or access denied"
            )

        try:
            # Create the run
            run = RunModel(
                experiment_id=experiment_id,
                name=req.name,
                description=req.description,
                status=RunStatusEnum.PENDING.value,
            )
            self.session.add(run)
            self.session.flush()  # Get the run ID

            # Create evaluations
            evaluations = []
            for eval_req in req.evaluations:
                evaluation = EvaluationModel(
                    run_id=run.id,
                    model_id=eval_req.model_id,
                    dataset_version_id=eval_req.dataset_version_id,
                    status=EvaluationStatusEnum.PENDING.value,
                    config=eval_req.config,
                )
                self.session.add(evaluation)
                evaluations.append(evaluation)

            self.session.commit()
            self.session.refresh(run)

            # Convert to schema with evaluations
            evaluation_schemas = [
                EvaluationWithResultsSchema(
                    id=eval.id,
                    run_id=eval.run_id,
                    model_id=eval.model_id,
                    dataset_version_id=eval.dataset_version_id,
                    status=EvaluationStatusEnum(eval.status),
                    config=eval.config,
                    metrics=[],
                    raw_results=None,
                )
                for eval in evaluations
            ]

            return RunWithEvaluationsSchema(
                id=run.id,
                experiment_id=run.experiment_id,
                name=run.name,
                description=run.description,
                status=RunStatusEnum(run.status),
                evaluations=evaluation_schemas,
            )

        except Exception as e:
            self.session.rollback()
            logger.debug(f"Failed to create run: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create run"
            ) from e

    def list_runs(self, experiment_id: uuid.UUID, user_id: uuid.UUID) -> List[RunSchema]:
        """List all runs for a given experiment.

        Parameters:
            experiment_id (uuid.UUID): ID of the experiment.
            user_id (uuid.UUID): ID of the user.

        Returns:
            List[RunSchema]: List of run schemas.

        Raises:
            HTTPException(status_code=404): If experiment not found or access denied.
        """
        ev = self.session.get(ExperimentModel, experiment_id)
        if not ev or ev.created_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experiment not found or access denied"
            )
        
        runs = (
            self.session.query(RunModel)
            .filter(
                RunModel.experiment_id == experiment_id,
                RunModel.status != RunStatusEnum.DELETED.value
            )
            .order_by(RunModel.created_at.desc())
            .all()
        )
        return [RunSchema.from_orm(r) for r in runs]

    def get_run_with_evaluations(self, run_id: uuid.UUID, user_id: uuid.UUID) -> RunWithEvaluationsSchema:
        """Get a run with its evaluations and results.

        Parameters:
            run_id (uuid.UUID): ID of the run.
            user_id (uuid.UUID): ID of the user.

        Returns:
            RunWithEvaluationsSchema: Run schema with evaluations.

        Raises:
            HTTPException(status_code=404): If run not found or access denied.
        """
        run = self.session.get(RunModel, run_id)
        if not run or run.experiment.created_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found or access denied"
            )

        # Get evaluations with metrics and results
        evaluations = (
            self.session.query(EvaluationModel)
            .filter(EvaluationModel.run_id == run_id)
            .all()
        )

        evaluation_schemas = []
        for eval in evaluations:
            # Get metrics for this evaluation
            metrics = (
                self.session.query(MetricModel)
                .filter(MetricModel.evaluation_id == eval.id)
                .all()
            )
            metrics_data = [
                {
                    "metric_name": metric.metric_name,
                    "mode": metric.mode,
                    "metric_value": float(metric.metric_value),
                }
                for metric in metrics
            ]

            # Get raw results for this evaluation
            raw_result = (
                self.session.query(RawResultModel)
                .filter(RawResultModel.evaluation_id == eval.id)
                .first()
            )
            raw_results_data = raw_result.preview_results if raw_result else None

            evaluation_schemas.append(
                EvaluationWithResultsSchema(
                    id=eval.id,
                    run_id=eval.run_id,
                    model_id=eval.model_id,
                    dataset_version_id=eval.dataset_version_id,
                    status=EvaluationStatusEnum(eval.status),
                    config=eval.config,
                    metrics=metrics_data,
                    raw_results=raw_results_data,
                )
            )

        return RunWithEvaluationsSchema(
            id=run.id,
            experiment_id=run.experiment_id,
            name=run.name,
            description=run.description,
            status=RunStatusEnum(run.status),
            evaluations=evaluation_schemas,
        )

    def update_run(
        self,
        run_id: uuid.UUID,
        req: UpdateRunRequest,
        user_id: uuid.UUID,
    ) -> RunSchema:
        """Update fields of an existing run.

        Parameters:
            run_id (uuid.UUID): ID of the run to update.
            req (UpdateRunRequest): Payload with optional fields to update.
            user_id (uuid.UUID): ID of the user attempting the update.

        Returns:
            RunSchema: Pydantic schema of the updated run.

        Raises:
            HTTPException(status_code=404): If run not found or access denied.
            HTTPException(status_code=500): If database update fails.
        """
        run = self.session.get(RunModel, run_id)
        if not run or run.experiment.created_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found or access denied"
            )

        if req.name is not None:
            run.name = req.name
        if req.description is not None:
            run.description = req.description
        if req.status is not None:
            run.status = req.status.value

        try:
            self.session.commit()
            self.session.refresh(run)
        except Exception as e:
            self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update run"
            ) from e
        return RunSchema.from_orm(run)

    def delete_run(self, run_id: uuid.UUID, user_id: uuid.UUID) -> None:
        """Soft-delete a run by marking its status as DELETED.

        Parameters:
            run_id (uuid.UUID): ID of the run to delete.
            user_id (uuid.UUID): ID of the user attempting the delete.

        Raises:
            HTTPException(status_code=404): If run not found or access denied.
            HTTPException(status_code=500): If database commit fails.
        """
        run = self.session.get(RunModel, run_id)
        if not run or run.experiment.created_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found or access denied"
            )
        run.status = RunStatusEnum.DELETED.value
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete run"
            ) from e

    # ------------------------ Evaluation Methods ------------------------

    def list_evaluations(self, run_id: uuid.UUID, user_id: uuid.UUID) -> List[EvaluationWithResultsSchema]:
        """List evaluations within a run.

        Parameters:
            run_id (uuid.UUID): ID of the run.
            user_id (uuid.UUID): ID of the user.

        Returns:
            List[EvaluationWithResultsSchema]: List of evaluations with results.

        Raises:
            HTTPException(status_code=404): If run not found or access denied.
        """
        run = self.session.get(RunModel, run_id)
        if not run or run.experiment.created_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found or access denied"
            )

        evaluations = (
            self.session.query(EvaluationModel)
            .filter(EvaluationModel.run_id == run_id)
            .all()
        )

        evaluation_schemas = []
        for eval in evaluations:
            # Get metrics for this evaluation
            metrics = (
                self.session.query(MetricModel)
                .filter(MetricModel.evaluation_id == eval.id)
                .all()
            )
            metrics_data = [
                {
                    "metric_name": metric.metric_name,
                    "mode": metric.mode,
                    "metric_value": float(metric.metric_value),
                }
                for metric in metrics
            ]

            # Get raw results for this evaluation
            raw_result = (
                self.session.query(RawResultModel)
                .filter(RawResultModel.evaluation_id == eval.id)
                .first()
            )
            raw_results_data = raw_result.preview_results if raw_result else None

            evaluation_schemas.append(
                EvaluationWithResultsSchema(
                    id=eval.id,
                    run_id=eval.run_id,
                    model_id=eval.model_id,
                    dataset_version_id=eval.dataset_version_id,
                    status=EvaluationStatusEnum(eval.status),
                    config=eval.config,
                    metrics=metrics_data,
                    raw_results=raw_results_data,
                )
            )

        return evaluation_schemas

    def get_evaluation_with_results(self, evaluation_id: uuid.UUID, user_id: uuid.UUID) -> EvaluationWithResultsSchema:
        """Get an evaluation with metrics and results.

        Parameters:
            evaluation_id (uuid.UUID): ID of the evaluation.
            user_id (uuid.UUID): ID of the user.

        Returns:
            EvaluationWithResultsSchema: Evaluation schema with results.

        Raises:
            HTTPException(status_code=404): If evaluation not found or access denied.
        """
        evaluation = self.session.get(EvaluationModel, evaluation_id)
        if not evaluation or evaluation.run.experiment.created_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found or access denied"
            )

        # Get metrics for this evaluation
        metrics = (
            self.session.query(MetricModel)
            .filter(MetricModel.evaluation_id == evaluation_id)
            .all()
        )
        metrics_data = [
            {
                "metric_name": metric.metric_name,
                "mode": metric.mode,
                "metric_value": float(metric.metric_value),
            }
            for metric in metrics
        ]

        # Get raw results for this evaluation
        raw_result = (
            self.session.query(RawResultModel)
            .filter(RawResultModel.evaluation_id == evaluation_id)
            .first()
        )
        raw_results_data = raw_result.preview_results if raw_result else None

        return EvaluationWithResultsSchema(
            id=evaluation.id,
            run_id=evaluation.run_id,
            model_id=evaluation.model_id,
            dataset_version_id=evaluation.dataset_version_id,
            status=EvaluationStatusEnum(evaluation.status),
            config=evaluation.config,
            metrics=metrics_data,
            raw_results=raw_results_data,
        )

    def update_evaluation(
        self,
        evaluation_id: uuid.UUID,
        req: UpdateEvaluationRequest,
        user_id: uuid.UUID,
    ) -> EvaluationSchema:
        """Update an evaluation.

        Parameters:
            evaluation_id (uuid.UUID): ID of the evaluation to update.
            req (UpdateEvaluationRequest): Payload with optional fields to update.
            user_id (uuid.UUID): ID of the user attempting the update.

        Returns:
            EvaluationSchema: Pydantic schema of the updated evaluation.

        Raises:
            HTTPException(status_code=404): If evaluation not found or access denied.
            HTTPException(status_code=500): If database update fails.
        """
        evaluation = self.session.get(EvaluationModel, evaluation_id)
        if not evaluation or evaluation.run.experiment.created_by != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation not found or access denied"
            )

        if req.status is not None:
            evaluation.status = req.status.value
        if req.config is not None:
            evaluation.config = req.config

        try:
            self.session.commit()
            self.session.refresh(evaluation)
        except Exception as e:
            self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update evaluation"
            ) from e
        return EvaluationSchema.from_orm(evaluation)

    # ------------------------ Dataset Methods (Keep existing) ------------------------

    def get_dataset_by_id(self, dataset_id: uuid.UUID) -> DatasetSchema:
        """Get a dataset by ID with associated traits.

        Parameters:
            dataset_id (uuid.UUID): ID of the dataset to retrieve.

        Returns:
            DatasetSchema: Pydantic schema of the dataset with traits.

        Raises:
            HTTPException(status_code=404): If dataset not found.
            HTTPException(status_code=500): If database query fails.
        """
        try:
            # Get the dataset
            dataset = self.session.get(DatasetModel, dataset_id)
            if not dataset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found"
                )

            # Get associated traits through pivot table
            traits_query = (
                self.session.query(TraitModel)
                .join(PivotModel, TraitModel.id == PivotModel.trait_id)
                .filter(PivotModel.dataset_id == dataset_id)
                .all()
            )

            # Convert traits to schema
            from budapp.eval_ops.schemas import DatasetBasic
            traits = [
                TraitSchema(
                    id=trait.id,
                    name=trait.name,
                    description=trait.description or "",
                    category="",
                    exps_ids=[],
                    datasets=[DatasetBasic(
                        id=dataset.id,
                        name=dataset.name,
                        description=dataset.description,
                        estimated_input_tokens=dataset.estimated_input_tokens,
                        estimated_output_tokens=dataset.estimated_output_tokens,
                        modalities=dataset.modalities,
                        sample_questions_answers=dataset.sample_questions_answers,
                        advantages_disadvantages=dataset.advantages_disadvantages,
                    )],  # This trait is associated with the current dataset
                )
                for trait in traits_query
            ]

            # Create dataset schema with traits
            dataset_schema = DatasetSchema(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                meta_links=dataset.meta_links,
                config_validation_schema=dataset.config_validation_schema,
                estimated_input_tokens=dataset.estimated_input_tokens,
                estimated_output_tokens=dataset.estimated_output_tokens,
                language=dataset.language,
                domains=dataset.domains,
                concepts=dataset.concepts,
                humans_vs_llm_qualifications=dataset.humans_vs_llm_qualifications,
                task_type=dataset.task_type,
                modalities=dataset.modalities,
                sample_questions_answers=dataset.sample_questions_answers,
                advantages_disadvantages=dataset.advantages_disadvantages,
                traits=traits,
            )

            return dataset_schema

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get dataset"
            ) from e

    def list_datasets(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Optional[DatasetFilter] = None,
    ) -> Tuple[List[DatasetSchema], int]:
        """List datasets with optional filtering and pagination.

        Parameters:
            offset (int): Number of records to skip (for pagination).
            limit (int): Maximum number of records to return.
            filters (Optional[DatasetFilter]): Filter parameters.

        Returns:
            Tuple[List[DatasetSchema], int]: A tuple of (list of DatasetSchema, total count).

        Raises:
            HTTPException(status_code=500): If database query fails.
        """
        try:
            q = self.session.query(DatasetModel)
            
            # Apply filters
            if filters:
                if filters.name:
                    q = q.filter(DatasetModel.name.ilike(f"%{filters.name}%"))
                if filters.modalities:
                    # Filter by modalities (JSONB contains any of the specified modalities)
                    for modality in filters.modalities:
                        q = q.filter(DatasetModel.modalities.contains([modality]))
                if filters.language:
                    # Filter by language (JSONB contains any of the specified languages)
                    for lang in filters.language:
                        q = q.filter(DatasetModel.language.contains([lang]))
                if filters.domains:
                    # Filter by domains (JSONB contains any of the specified domains)
                    for domain in filters.domains:
                        q = q.filter(DatasetModel.domains.contains([domain]))

            # Get total count before applying pagination
            total_count = q.count()
            
            # Apply pagination and get results
            datasets = q.offset(offset).limit(limit).all()
            
            # For each dataset, get associated traits
            dataset_schemas = []
            for dataset in datasets:
                # Get traits associated with this dataset
                traits_query = (
                    self.session.query(TraitModel)
                    .join(PivotModel, TraitModel.id == PivotModel.trait_id)
                    .filter(PivotModel.dataset_id == dataset.id)
                    .all()
                )
                
                # Convert traits to schema (simplified for list view)
                from budapp.eval_ops.schemas import DatasetBasic
                traits = [
                    TraitSchema(
                        id=trait.id,
                        name=trait.name,
                        description=trait.description or "",
                        category="",
                        exps_ids=[],
                        datasets=[],  # Don't include datasets in list view to avoid circular references
                    )
                    for trait in traits_query
                ]
                
                dataset_schema = DatasetSchema(
                    id=dataset.id,
                    name=dataset.name,
                    description=dataset.description,
                    meta_links=dataset.meta_links,
                    config_validation_schema=dataset.config_validation_schema,
                    estimated_input_tokens=dataset.estimated_input_tokens,
                    estimated_output_tokens=dataset.estimated_output_tokens,
                    language=dataset.language,
                    domains=dataset.domains,
                    concepts=dataset.concepts,
                    humans_vs_llm_qualifications=dataset.humans_vs_llm_qualifications,
                    task_type=dataset.task_type,
                    modalities=dataset.modalities,
                    sample_questions_answers=dataset.sample_questions_answers,
                    advantages_disadvantages=dataset.advantages_disadvantages,
                    traits=traits,
                )
                dataset_schemas.append(dataset_schema)
            
            return dataset_schemas, total_count

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list datasets"
            ) from e

    def create_dataset(self, req: CreateDatasetRequest) -> DatasetSchema:
        """Create a new dataset with traits.

        Parameters:
            req (CreateDatasetRequest): Payload containing dataset information and trait associations.

        Returns:
            DatasetSchema: Pydantic schema of the created dataset with traits.

        Raises:
            HTTPException(status_code=400): If validation fails.
            HTTPException(status_code=500): If database insertion fails.
        """
        try:
            # Create the dataset
            dataset = DatasetModel(
                name=req.name,
                description=req.description,
                meta_links=req.meta_links,
                config_validation_schema=req.config_validation_schema,
                estimated_input_tokens=req.estimated_input_tokens,
                estimated_output_tokens=req.estimated_output_tokens,
                language=req.language,
                domains=req.domains,
                concepts=req.concepts,
                humans_vs_llm_qualifications=req.humans_vs_llm_qualifications,
                task_type=req.task_type,
                modalities=req.modalities,
                sample_questions_answers=req.sample_questions_answers,
                advantages_disadvantages=req.advantages_disadvantages,
            )
            self.session.add(dataset)
            self.session.flush()  # Get the dataset ID

            # Create trait associations
            if req.trait_ids:
                for trait_id in req.trait_ids:
                    # Verify trait exists
                    trait = self.session.get(TraitModel, trait_id)
                    if not trait:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Trait with ID {trait_id} not found"
                        )
                    
                    pivot = PivotModel(trait_id=trait_id, dataset_id=dataset.id)
                    self.session.add(pivot)

            self.session.commit()
            self.session.refresh(dataset)

            # Return the created dataset with traits
            return self.get_dataset_by_id(dataset.id)

        except HTTPException:
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            logger.debug(f"Failed to create dataset: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create dataset"
            ) from e

    def update_dataset(self, dataset_id: uuid.UUID, req: UpdateDatasetRequest) -> DatasetSchema:
        """Update an existing dataset and its traits.

        Parameters:
            dataset_id (uuid.UUID): ID of the dataset to update.
            req (UpdateDatasetRequest): Payload with optional dataset fields and trait associations.

        Returns:
            DatasetSchema: Pydantic schema of the updated dataset with traits.

        Raises:
            HTTPException(status_code=404): If dataset not found.
            HTTPException(status_code=400): If validation fails.
            HTTPException(status_code=500): If database update fails.
        """
        try:
            dataset = self.session.get(DatasetModel, dataset_id)
            if not dataset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found"
                )

            # Update dataset fields
            if req.name is not None:
                dataset.name = req.name
            if req.description is not None:
                dataset.description = req.description
            if req.meta_links is not None:
                dataset.meta_links = req.meta_links
            if req.config_validation_schema is not None:
                dataset.config_validation_schema = req.config_validation_schema
            if req.estimated_input_tokens is not None:
                dataset.estimated_input_tokens = req.estimated_input_tokens
            if req.estimated_output_tokens is not None:
                dataset.estimated_output_tokens = req.estimated_output_tokens
            if req.language is not None:
                dataset.language = req.language
            if req.domains is not None:
                dataset.domains = req.domains
            if req.concepts is not None:
                dataset.concepts = req.concepts
            if req.humans_vs_llm_qualifications is not None:
                dataset.humans_vs_llm_qualifications = req.humans_vs_llm_qualifications
            if req.task_type is not None:
                dataset.task_type = req.task_type
            if req.modalities is not None:
                dataset.modalities = req.modalities
            if req.sample_questions_answers is not None:
                dataset.sample_questions_answers = req.sample_questions_answers
            if req.advantages_disadvantages is not None:
                dataset.advantages_disadvantages = req.advantages_disadvantages

            # Update trait associations if provided
            if req.trait_ids is not None:
                # Remove existing associations
                self.session.query(PivotModel).filter(
                    PivotModel.dataset_id == dataset_id
                ).delete()

                # Add new associations
                for trait_id in req.trait_ids:
                    # Verify trait exists
                    trait = self.session.get(TraitModel, trait_id)
                    if not trait:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Trait with ID {trait_id} not found"
                        )
                    
                    pivot = PivotModel(trait_id=trait_id, dataset_id=dataset_id)
                    self.session.add(pivot)

            self.session.commit()
            self.session.refresh(dataset)

            # Return the updated dataset with traits
            return self.get_dataset_by_id(dataset_id)

        except HTTPException:
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            logger.debug(f"Failed to update dataset: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update dataset"
            ) from e

    def delete_dataset(self, dataset_id: uuid.UUID) -> None:
        """Delete a dataset and its trait associations.

        Parameters:
            dataset_id (uuid.UUID): ID of the dataset to delete.

        Raises:
            HTTPException(status_code=404): If dataset not found.
            HTTPException(status_code=500): If database deletion fails.
        """
        try:
            dataset = self.session.get(DatasetModel, dataset_id)
            if not dataset:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Dataset not found"
                )

            # Delete trait associations first
            self.session.query(PivotModel).filter(
                PivotModel.dataset_id == dataset_id
            ).delete()

            # Delete the dataset
            self.session.delete(dataset)
            self.session.commit()

        except HTTPException:
            self.session.rollback()
            raise
        except Exception as e:
            self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete dataset"
            ) from e
