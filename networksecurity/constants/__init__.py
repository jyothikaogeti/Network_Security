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
SCHEMA_FILE_PATH = "schema.yaml"

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


"""
Data Validation related constants start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR:str = "validated_data"
DATA_VALIDATION_INVALID_DIR:str = "invalid_data"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = "data_drift_report.yaml"