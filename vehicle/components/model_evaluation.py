

import sys
import torch
from torch.utils.data import DataLoader
from vehicle.ml.models.model_evaluation import evaluate
from vehicle.constants import DEVICE
from vehicle.logger import logging
from vehicle.exception import VehicleException
from vehicle.utils.main_utils import load_object
from vehicle.entity.config_entity import ModelEvaluationConfig, ModelTrainerConfig
from vehicle.entity.artifacts_entity import ModelTrainerArtifacts, DataTransformationArtifacts, ModelEvaluationArtifacts


class ModelEvaluation:

    def __init__(self, model_evaluation_config=ModelEvaluationConfig,
                 model_trainer_config= ModelTrainerConfig,
                data_transformation_artifacts=DataTransformationArtifacts,
                model_trainer_artifacts=ModelTrainerArtifacts):

        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_artifacts = model_trainer_artifacts

    def collate_fn(batch):
        """
        This is our collating function for the train dataloader,
        it allows us to create batches of data that can be easily pass into the model
        """
        try:
            return tuple(zip(*batch))
        except Exception as e:
            raise VehicleException(e, sys) from e


    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
                Method Name :   initiate_model_evaluation
                Description :   This function is used to initiate all steps of the model evaluation

                Output      :   Returns model evaluation artifact
                On Failure  :   Write an exception log and then raise an exception
        """

        try:
            model = torch.load(self.model_trainer_artifacts.trained_model_path)

            test_dataset = load_object(self.data_transformation_artifacts.transformed_test_object)

            test_loader = DataLoader(test_dataset,
                                      batch_size=self.model_trainer_config.BATCH_SIZE,
                                      shuffle=self.model_trainer_config.SHUFFLE,
                                      num_workers=self.model_trainer_config.NUM_WORKERS,
                                      collate_fn=self.collate_fn
                                      )

            logging.info("loaded saved model")

            model = model.to(DEVICE)

            # evaluate(model, test_loader, device=DEVICE)



        except Exception as e:
            raise VehicleException(e, sys) from e

