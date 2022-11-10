
import sys
import tqdm
import math
import numpy as np
import pandas as pd
import torch
from torchvision import models
from vehicle.constants import DEVICE
from vehicle.logger import logging
from vehicle.exception import VehicleException
from vehicle.utils.main_utils import load_object
from vehicle.ml.models.model_optimiser import model_optimiser
from vehicle.entity.config_entity import ModelTrainerConfig
from vehicle.entity.artifacts_entity import DataTransformationArtifacts, ModelTrainerArtifacts

class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts,
                    model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config


    def train_one_epoch(self, model, optimizer, loader, device, epoch):
        try:
            model.to(device)
            model.train() 
            all_losses = []
            all_losses_dict = []
            
            for images, targets in tqdm(loader):
                images = list(image.to(device) for image in images)
                targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets) # the model computes the loss automatically if we pass in targets
                losses = sum(loss for loss in loss_dict.values())
                loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
                loss_value = losses.item()
                
                all_losses.append(loss_value)
                all_losses_dict.append(loss_dict_append)
                
                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping trainig") # train if loss becomes infinity
                    print(loss_dict)
                    sys.exit(1)
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                
            all_losses_dict = pd.DataFrame(all_losses_dict) # for printing
            print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f}, loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
                epoch, optimizer.param_groups[0]['lr'], np.mean(all_losses),
                all_losses_dict['loss_classifier'].mean(),
                all_losses_dict['loss_box_reg'].mean(),
                all_losses_dict['loss_rpn_box_reg'].mean(),
                all_losses_dict['loss_objectness'].mean()
            ))

        except Exception as e:
            raise VehicleException(e, sys) from e




    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:

            train_loader = load_object(self.data_transformation_artifacts.transformed_train_object)

            logging.info("Loaded training data loader object")

            model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

            logging.info("Loaded faster rcnn  model")

            in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head

            model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, self.data_transformation_artifacts.number_of_classes)

            optimiser = model_optimiser(model)

            for epoch in range(self.model_trainer_config.EPOCH):
                self.train_one_epoch(model, optimiser, train_loader, self.model_trainer_config.DEVICE, epoch)


        except Exception as e:
            raise VehicleException(e, sys) from e



