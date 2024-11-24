import os
import sys
import pandas as pd
import numpy as np
import pymongo
from typing import List
import pymongo.mongo_client
from sklearn.model_selection import train_test_split
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def read_data_from_mongodb(self):
        """
        Read data from MOngoDB
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            network_data = pd.DataFrame(list(collection.find()))
            if "_id" in network_data.columns.to_list():
                network_data = network_data.drop(columns=["_id"], axis=1)

            network_data.replace({"na":np.nan}, inplace=True)
            return network_data

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
        
    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # Creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")
            logging.info("Exited split_data_as_train_test method of Data_Ingestion class")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Exporting train and test file path")

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info(f"Exported train and test file path")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

            
    def initialize_data_ingestion(self):
        try:
            df = self.read_data_from_mongodb()
            df = self.export_data_into_feature_store(dataframe=df)
            df = self.split_data_as_train_test(dataframe=df)

            data_ingestion_artifact = DataIngestionArtifact(
              trained_file_path=self.data_ingestion_config.training_file_path,
              test_file_path=self.data_ingestion_config.testing_file_path
            )
            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
