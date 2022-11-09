import os
import sys
from zipfile import Path, ZipFile
from vehicle.entity.config_entity import DataIngestionConfig
from vehicle.entity.artifacts_entity import DataIngestionArtifacts
from vehicle.configuration.s3_operations import S3Operation
from vehicle.exception import VehicleException
from vehicle.logger import logging
from vehicle.constants import *
from PIL import Image
import numpy as np
import shutil


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, s3_operations: S3Operation):
        self.data_ingestion_config = data_ingestion_config
        self.s3_operations = s3_operations


    def get_data_from_s3(self) -> None:
        try:
            logging.info("Entered the get_data_from_s3 method of Data ingestion class")
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            self.s3_operations.read_data_from_s3(self.data_ingestion_config.ZIP_FILE_NAME,self.data_ingestion_config.BUCKET_NAME,
                                                self.data_ingestion_config.ZIP_FILE_PATH)
            logging.info("Exited the get_data_from_s3 method of Data ingestion class")
        except Exception as e:
            raise VehicleException(e, sys) from e

    def unzip_and_clean(self) -> None:
        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try:
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)
            logging.info("Exited the unzip_and_clean method of Data ingestion class")
        except Exception as e:
            raise VehicleException(e, sys) from e

    def train_test_split(self) -> Path:
        """
        This function would split the raw data into train and test folder
        """
        logging.info("Entered the train_test_split method of Data ingestion class")
        try:
            # 1. make train and test folder 
            unzipped_images = self.data_ingestion_config.UNZIPPED_FILE_PATH

            os.makedirs(self.data_ingestion_config.TRAIN_DATA_ARTIFACT_DIR, exist_ok=True)
            os.makedirs(self.data_ingestion_config.TEST_DATA_ARTIFACT_DIR, exist_ok=True)
            logging.info("Created train data artifacts and test data artifacts directories")
            #params.yaml
            test_ratio = self.data_ingestion_config.PARAMS_TEST_RATIO

            #print(train_path)
            classes_dir = [CLASS_LABEL_1, CLASS_LABEL_2]

            for cls in classes_dir:
                os.makedirs(os.path.join(self.data_ingestion_config.TRAIN_DATA_ARTIFACT_DIR, cls), exist_ok= True)
                os.makedirs(os.path.join(self.data_ingestion_config.TEST_DATA_ARTIFACT_DIR, cls), exist_ok=True)
                logging.info("Created train data artifacts and test data artifacts directories with class")
            # 2. Split the raw data
            raw_data_path = os.path.join(unzipped_images)

            for cls in classes_dir:
                all_file_names = os.listdir(os.path.join(raw_data_path, cls))
 
                np.random.shuffle(all_file_names)
                train_file_name, test_file_name = np.split(np.array(all_file_names),
                                    [int(len(all_file_names)* (1 - test_ratio))])

                train_names = [os.path.join(raw_data_path, cls, name) for name in train_file_name.tolist()]
                test_names = [os.path.join(raw_data_path, cls, name) for name in test_file_name.tolist()]

                for name in train_names:
                    shutil.copy(name, os.path.join(self.data_ingestion_config.TRAIN_DATA_ARTIFACT_DIR, cls))

                for name in test_names:
                    shutil.copy(name, os.path.join(self.data_ingestion_config.TEST_DATA_ARTIFACT_DIR, cls))   

            shutil.rmtree(self.data_ingestion_config.UNZIPPED_FILE_PATH, ignore_errors=True)
            logging.info("Exited the train_test_split method of Data ingestion class")

            return self.data_ingestion_config.TRAIN_DATA_ARTIFACT_DIR, self.data_ingestion_config.TEST_DATA_ARTIFACT_DIR

        except Exception as e:
            raise VehicleException(e, sys) from e


    def initiate_data_ingestion(self) -> DataIngestionArtifacts: 
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")
        try: 
            self.get_data_from_s3()
            self.unzip_and_clean()
            # train_file_path, test_file_path = self.train_test_split()
            # data_ingestion_artifact = DataIngestionArtifacts(train_file_path=train_file_path, 
            #                                                     test_file_path=test_file_path)
            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")

            # return data_ingestion_artifact

        except Exception as e:
            raise VehicleException(e, sys) from e