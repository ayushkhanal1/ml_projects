import sys
import os
from src.exception import CustomException
from src.logger import logging  
 
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import accuracy_score
from src.utils import save_objec
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor



@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')  # Path to save the trained model

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array,preprocessor_path):
        try:
            logging.info("Model Trainer started")
            # Splitting input and target features
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNeighbors Regressor": KNeighborsRegressor()
            }

            best_model_name = None
            best_model_score = -float("inf")
            best_model = None
            model_report = {}

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2_score_value = r2_score(y_test, y_pred)
                model_report[model_name] = r2_score
                logging.info(f"{model_name} R2 Score: {r2_score}")

                if r2_score_value > best_model_score:
                    best_model_score = r2_score_value
                    best_model_name = model_name
                    best_model = model

            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score}")

            # Save the best model to the specified file path
            save_objec(
                file_path=self.model_trainer_config.trained_model_file_path,
                object=best_model
            )

            logging.info("Model Trainer completed")

            return best_model_name, best_model_score

        except Exception as e:
            logging.error("Error occurred in Model Trainer")
            raise CustomException(e, sys)