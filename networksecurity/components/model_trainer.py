import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.common import save_pickle_file, load_object, load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metrics import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.metrics import r2_score

class ModelTrainer:
    def __init__(self,
                 model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    
    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Random Forest":RandomForestClassifier(verbose=1),
            "Decision Tree":DecisionTreeClassifier(),
            "Gradient Boosting":GradientBoostingClassifier(verbose=1),
            "Logistic Regression":LogisticRegression(verbose=1),
            "AdaBoost":AdaBoostClassifier()
                } 
        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy"],  # Determines how the split is evaluated.
                "max_depth": [None, 10, 20],  # Controls overfitting.
                "min_samples_split": [2, 10]  # Minimum samples required to split.
            },
            "Random Forest": {
                "n_estimators": [50, 100, 200],  # Number of trees in the forest.
                "max_depth": [None, 10, 20],  # Depth of each tree to prevent overfitting.
                "max_features": ["sqrt", "log2"],  # Features considered for best split.
            },
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01],  # Step size for updating weights.
                "n_estimators": [50, 100, 200],  # Number of boosting stages.
                "max_depth": [3, 5],  # Depth of the individual trees.
            },
            "Logistic Regression": {
                "penalty": ["l2"],  # Regularization technique (L2 is the most common).
                "C": [0.1, 1, 10],  # Inverse of regularization strength.
                "solver": ["lbfgs", "liblinear"],  # Optimization algorithm.
            },
            "AdaBoost": {
                "n_estimators": [50, 100, 200],  # Number of weak learners.
                "learning_rate": [0.1, 1],  # Shrinks the contribution of each classifier.
            }
        }


        model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, 
                                            X_test=X_test, y_test=y_test,
                                            models=models,params=params) 
        
        # To get the best model score from dict
        best_model_score = max(sorted(model_report.values()))

        # To get best model name from dict
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_pickle_file(self.model_trainer_config.trained_model_file_path, obj=Network_Model)

        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                 train_metric_artifact=classification_train_metric,
                                 test_metric_artifact=classification_test_metric)

        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    
    def initialize_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)