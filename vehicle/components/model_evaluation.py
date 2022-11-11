

import sys
import torch
from vehicle.constants import DEVICE
from vehicle.logger import logging
from vehicle.exception import VehicleException
from vehicle.entity.config_entity import ModelEvaluationConfig
from vehicle.entity.artifacts_entity import ModelTrainerArtifacts, DataTransformationArtifacts, ModelEvaluationArtifacts


class ModelEvaluation:

    def __init__(self, model_evaluation_config=ModelEvaluationConfig,
                data_transformation_artifacts=DataTransformationArtifacts,
                model_trainer_artifacts=ModelTrainerArtifacts):

        self.model_evaluation_config = model_evaluation_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_artifacts = model_trainer_artifacts


    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
                Method Name :   initiate_model_evaluation
                Description :   This function is used to initiate all steps of the model evaluation

                Output      :   Returns model evaluation artifact
                On Failure  :   Write an exception log and then raise an exception
        """

        try:
            model = torch.load(self.model_trainer_artifacts.trained_model_path)

            logging.info("loaded saved model")

            model = model.to(DEVICE)

        except Exception as e:
            raise VehicleException(e, sys) from e

