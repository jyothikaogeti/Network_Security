from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig, TrainingPipelineConfig
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiate the Data Ingestion")
        data_ingestion_artifact = data_ingestion.initialize_data_ingestion()
        logging.info("Data Initialization Completed")
        print(data_ingestion_artifact)

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Initiate the Data Validation")
        data_validation_artifact = data_validation.initialize_data_validation()
        logging.info("Data Validation Completed")
        print(data_validation_artifact)

        data_transformation_config=DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info("Initiate Data Transformation")
        data_transformation_artifact=data_transformation.initialize_data_transformation()
        logging.info("Data Transformation Completed")
        print(data_transformation_artifact)

        model_trainer_config=ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        logging.info("Model Training Started")
        model_trainer_artifact=model_trainer.initialize_model_trainer()
        logging.info("Model Training Completed")
        print(model_trainer_artifact)

    except Exception as e:
        raise NetworkSecurityException(e, SystemError)

