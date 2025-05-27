import uuid
from typing import List, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.eval_ops.models import (
    EvaluationRunStatusEnum,
)
from budapp.eval_ops.models import Experiment as ExperimentModel
from budapp.eval_ops.models import ExpTrait as TraitModel
from budapp.eval_ops.models import ExpDataset as DatasetModel
from budapp.eval_ops.models import ExpTraitsDatasetPivot as PivotModel
from budapp.eval_ops.models import (
    Run as RunModel,
)
from budapp.eval_ops.schemas import (
    CreateExperimentRequest,
    CreateRunRequest,
    UpdateExperimentRequest,
    UpdateRunRequest,
    CreateDatasetRequest,
    UpdateDatasetRequest,
    DatasetFilter,
)
from budapp.eval_ops.schemas import (
    Experiment as ExperimentSchema,
)
from budapp.eval_ops.schemas import (
    Run as RunSchema,
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
        """Paginate and filter Trait records.

        Parameters:
            offset (int): Number of records to skip (for pagination).
            limit (int): Maximum number of records to return.
            name (Optional[str]): Case-insensitive substring filter on trait name.
            unique_id (Optional[str]): Exact UUID filter on trait ID.

        Returns:
            Tuple[List[TraitSchema], int]: A tuple of (list of TraitSchema, total count).

        Raises:
            HTTPException(status_code=500): If database query fails.
        """
        try:
            q = self.session.query(TraitModel)
            if name:
                q = q.filter(TraitModel.name.ilike(f"%{name}%"))
            if unique_id:
                q = q.filter(TraitModel.id == uuid.UUID(unique_id))
            total = q.count()
            traits = (
                q.order_by(TraitModel.created_at)
                 .offset(offset)
                 .limit(limit)
                 .all()
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list traits"
            ) from e

        # Get datasets for each trait
        result = []
        for t in traits:
            # Get associated datasets through pivot table
            datasets_query = (
                self.session.query(DatasetModel)
                .join(PivotModel, DatasetModel.id == PivotModel.dataset_id)
                .filter(PivotModel.trait_id == t.id)
                .all()
            )
            
            # Convert to DatasetBasic schema
            from budapp.eval_ops.schemas import DatasetBasic
            datasets = [
                DatasetBasic(
                    id=dataset.id,
                    name=dataset.name,
                    description=dataset.description,
                    estimated_input_tokens=dataset.estimated_input_tokens,
                    estimated_output_tokens=dataset.estimated_output_tokens,
                    modalities=dataset.modalities,
                )
                for dataset in datasets_query
            ]
            
            result.append(TraitSchema(
                id=t.id,
                name=t.name,
                description=t.description or "",
                category="",
                exps_ids=[],
                datasets=datasets,
            ))
        
        return result, total

    def create_run(self, experiment_id: uuid.UUID, req: CreateRunRequest, user_id: uuid.UUID) -> RunSchema:
        """Create a new run under the given experiment.

        Raises 404 if experiment not found or access denied.
        """
        ev = self.session.get(ExperimentModel, experiment_id)
        if not ev or ev.created_by != user_id or ev.status == EvaluationRunStatusEnum.DELETED.value:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Experiment not found or access denied")
        run = RunModel(
            experiment_id=experiment_id,
            status=EvaluationRunStatusEnum.PENDING.value,
        )
        try:
            self.session.add(run)
            self.session.commit()
            self.session.refresh(run)
        except Exception as e:
            self.session.rollback()
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to create run") from e
        return RunSchema.from_orm(run)

    def list_runs(self, experiment_id: uuid.UUID, user_id: uuid.UUID) -> List[RunSchema]:
        """List all runs for a given experiment.

        Raises 404 if experiment not found or access denied.
        """
        ev = self.session.get(ExperimentModel, experiment_id)
        if not ev or ev.created_by != user_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Experiment not found or access denied")
        runs = (
            self.session.query(RunModel)
            .filter(RunModel.experiment_id == experiment_id)
            .order_by(RunModel.created_at.desc())
            .all()
        )
        return [RunSchema.from_orm(r) for r in runs]

    def update_run(
        self,
        run_id: uuid.UUID,
        req: UpdateRunRequest,
        user_id: uuid.UUID,
    ) -> RunSchema:
        """Update fields of an existing run. 404 if not found/access denied.

        Raises 404 if run not found.
        """
        run = self.session.get(RunModel, run_id)
        if not run:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
        # check ownership via parent experiment
        if run.experiment.created_by != user_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Access denied")
        if req.status is not None:
            run.status = req.status.value
        try:
            self.session.commit()
            self.session.refresh(run)
        except Exception as e:
            self.session.rollback()
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to update run") from e
        return RunSchema.from_orm(run)

    def delete_run(self, run_id: uuid.UUID, user_id: uuid.UUID) -> None:
        """Soft-delete a run by marking its status as DELETED. 404 if not found/access denied.

        Raises 404 if run not found.
        """
        run = self.session.get(RunModel, run_id)
        if not run or run.experiment.created_by != user_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found or access denied")
        run.status = EvaluationRunStatusEnum.DELETED.value
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to delete run") from e

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

            total = q.count()
            datasets = (
                q.order_by(DatasetModel.created_at.desc())
                 .offset(offset)
                 .limit(limit)
                 .all()
            )

            # Get traits for each dataset
            result = []
            for dataset in datasets:
                traits_query = (
                    self.session.query(TraitModel)
                    .join(PivotModel, TraitModel.id == PivotModel.trait_id)
                    .filter(PivotModel.dataset_id == dataset.id)
                    .all()
                )

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
                        )],  # This trait is associated with the current dataset
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
                    traits=traits,
                )
                result.append(dataset_schema)

            return result, total

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list datasets"
            ) from e

    def create_dataset(self, req: CreateDatasetRequest) -> DatasetSchema:
        """Create a new dataset with associated traits.

        Parameters:
            req (CreateDatasetRequest): Payload containing dataset information.

        Returns:
            DatasetSchema: Pydantic schema of the created dataset with traits.

        Raises:
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
            )

            self.session.add(dataset)
            self.session.flush()  # To get the dataset ID

            # Create trait associations
            traits = []
            if req.trait_ids:
                for trait_id in req.trait_ids:
                    # Verify trait exists
                    trait = self.session.get(TraitModel, trait_id)
                    if trait:
                        pivot = PivotModel(trait_id=trait_id, dataset_id=dataset.id)
                        self.session.add(pivot)
                        from budapp.eval_ops.schemas import DatasetBasic
                        traits.append(TraitSchema(
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
                            )],  # This trait is associated with the current dataset
                        ))

            self.session.commit()
            self.session.refresh(dataset)

            # Return dataset schema with traits
            return DatasetSchema(
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
                traits=traits,
            )

        except Exception as e:
            self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create dataset"
            ) from e

    def update_dataset(self, dataset_id: uuid.UUID, req: UpdateDatasetRequest) -> DatasetSchema:
        """Update an existing dataset and its traits.

        Parameters:
            dataset_id (uuid.UUID): ID of the dataset to update.
            req (UpdateDatasetRequest): Payload with updated dataset information.

        Returns:
            DatasetSchema: Pydantic schema of the updated dataset with traits.

        Raises:
            HTTPException(status_code=404): If dataset not found.
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

            # Update trait associations if provided
            if req.trait_ids is not None:
                # Remove existing associations
                self.session.query(PivotModel).filter(PivotModel.dataset_id == dataset_id).delete()
                
                # Add new associations
                for trait_id in req.trait_ids:
                    trait = self.session.get(TraitModel, trait_id)
                    if trait:
                        pivot = PivotModel(trait_id=trait_id, dataset_id=dataset_id)
                        self.session.add(pivot)

            self.session.commit()
            self.session.refresh(dataset)

            # Get updated traits
            traits_query = (
                self.session.query(TraitModel)
                .join(PivotModel, TraitModel.id == PivotModel.trait_id)
                .filter(PivotModel.dataset_id == dataset_id)
                .all()
            )

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
                    )],  # This trait is associated with the current dataset
                )
                for trait in traits_query
            ]

            return DatasetSchema(
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
                traits=traits,
            )

        except HTTPException:
            raise
        except Exception as e:
            self.session.rollback()
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
            self.session.query(PivotModel).filter(PivotModel.dataset_id == dataset_id).delete()
            
            # Delete the dataset
            self.session.delete(dataset)
            self.session.commit()

        except HTTPException:
            raise
        except Exception as e:
            self.session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete dataset"
            ) from e
