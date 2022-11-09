# from vehicle.models.model import Net 
import os
import torch
from torchsummary import summary
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data Ingestion Constants
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
LOGS_DIR = 'logs'
LOGS_FILE_NAME = 'vehicle.log'

BUCKET_NAME = 'vehicle-object-detection'
ZIP_FILE_NAME = 'Vehicle-dataset.zip'  

CLASS_LABEL_1 = 'vehicles'
CLASS_LABEL_2 = 'Ambulance'
CLASS_LABEL_3 = 'Bus'
CLASS_LABEL_4 = 'Car'
CLASS_LABEL_5 = 'Motorcycle'
CLASS_LABEL_6 = 'Truck'

RAW_FILE_NAME = 'vehicle'

DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'
DATA_INGESTION_TRAIN_DIR = 'Train'
DATA_INGESTION_TEST_DIR = 'Test'
PARAMS_TEST_RATIO = 0.2

# Data transformation constants 
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
DATA_TRANSFORMATION_TRAIN_DIR = 'Train'
DATA_TRANSFORMATION_TEST_DIR = 'Test'

BRIGHTNESS = 0.10
CONTRAST = 0.1
SATURATION = 0.10
HUE = 0.1
RESIZE = 224
CENTERCROP = 224
RANDOMROTATION = 10
NORMALIZE_LIST_1 = [0.485, 0.456, 0.406]
NORMALIZE_LIST_2 = [0.229, 0.224, 0.225]

BATCH_SIZE = 2
SHUFFLE = False
PIN_MEMORY= True

# Model Training Constants 
TRAINED_MODEL_DIR = 'TrainedModel'
TRAINED_MODEL_NAME = 'model.pt'
# MODEL = Net()
use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")
# OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=0.01, momentum=0.8)

PARAMS_EPOCHS = 3
STEP_SIZE = 6
GAMMA = 0.5
EPOCH = 1

APP_HOST = "0.0.0.0"
APP_PORT = 8001

# Prediction Constants
PREDICTION_LABEL = {"0" : CLASS_LABEL_1, "1" : CLASS_LABEL_2, "2" : CLASS_LABEL_3, "3" : CLASS_LABEL_4, "4" : CLASS_LABEL_5, "5" : CLASS_LABEL_6}

# AWS CONSTANTS
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "ap-south-1"