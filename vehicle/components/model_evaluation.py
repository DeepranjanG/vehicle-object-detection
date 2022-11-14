

import sys
import torch
import math
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
# from vehicle.ml.detection.engine import evaluate
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

    @staticmethod
    def collate_fn(batch):
        """
        This is our collating function for the train dataloader,
        it allows us to create batches of data that can be easily pass into the model
        """
        try:
            return tuple(zip(*batch))
        except Exception as e:
            raise VehicleException(e, sys) from e

    def evaluate(self, model, dataloader, device):
        try:

            model.eval()

            running_loss = 0.0
            loss_value = 0.0
            losses = 0.0

            all_losses = []
            all_losses_dict = []

            for images, targets in tqdm(dataloader):
                images = list(img.to(device) for img in images)
                targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

                with torch.no_grad():
                    loss_dict = model(images, targets)

                    # this returned object from the model:
                    # len is 4 (so index here), which is probably because of the size of the batch
                    # loss_dict[index]['boxes']
                    # loss_dict[index]['labels']
                    # loss_dict[index]['scores']
                #     for x in range(1):
                #         loss_value += sum(loss for loss in loss_dict[x]['scores'])
                #
                # running_loss += loss_value

                    print(loss_dict)

                    losses += sum(loss for loss in loss_dict.values())
                    loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
                    loss_value = losses.item()

                    all_losses.append(loss_value)
                    all_losses_dict.append(loss_dict_append)

                    if not math.isfinite(loss_value):
                        print(f"Loss is {loss_value}, stopping training")  # train if loss becomes infinity
                        print(loss_dict)
                        sys.exit(1)

                    losses.backward()
                all_losses_dict = pd.DataFrame(all_losses_dict)  # for printing

                print(
                    "loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
                        all_losses_dict['loss_classifier'].mean(),
                        all_losses_dict['loss_box_reg'].mean(),
                        all_losses_dict['loss_rpn_box_reg'].mean(),
                        all_losses_dict['loss_objectness'].mean()
                    ))

            return all_losses_dict
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
            # model = torch.load(self.model_trainer_artifacts.trained_model_path)
            model = torch.load(r"D:\Project\DL\torch-object-detection\artifacts\PredictModel\model.pt")

            test_dataset = load_object(self.data_transformation_artifacts.transformed_test_object)

            test_loader = DataLoader(test_dataset,
                                      batch_size=self.model_trainer_config.BATCH_SIZE,
                                      shuffle=self.model_trainer_config.SHUFFLE,
                                      num_workers=self.model_trainer_config.NUM_WORKERS,
                                      collate_fn=self.collate_fn
                                      )

            logging.info("loaded saved model")

            model = model.to(DEVICE)

            loss = self.evaluate(model, test_loader, device=DEVICE)

            print(loss)

        except Exception as e:
            raise VehicleException(e, sys) from e

