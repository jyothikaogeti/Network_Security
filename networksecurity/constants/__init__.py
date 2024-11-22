import os
import sys
import pandas as pd
import numpy as np

"""
Defining common constant variable for training pipeline
"""
TARGET_COLUMN = "Result"
PIPELINE_NAME: str = "NetworkSecurity"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "phisingData.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"


"""
Data Ingestion related constants start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLESTION_NAME: str = "NetworkData"
DATA_INGESTION_DATABASE_NAME: str = "NETWORK"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested_data"
DATA_INGESTION_TRAIN_TEST_TEST_SPLIT_RATIO: float = 0.2