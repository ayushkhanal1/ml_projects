import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_objec  

@dataclass    
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')  # Path to save the preprocessor object


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")

            # Define which columns are numerical and which are categorical
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']


            # Create a pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
                    ('scaler', StandardScaler())  # Standardize numerical features
                ]
            )

            # Create a pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one hot encoder',OneHotEncoder()),  # Fill missing values with most frequent value
                    ('scaler', StandardScaler(with_mean=False))  # Standardize categorical features
                ]
            )

            logging.info("Numerical and categorical pipelines created")

            # Combine both pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessor object created successfully")

            return preprocessor
        
        except Exception as e:
            logging.error("Error in get_data_transformer_object")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            # Separate input features and target variable for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target variable for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply the preprocessor to the training and testing input features
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessing object on training and testing datasets.")

            # Combine the transformed features with the target variable
            # Convert the target pandas Series to a 1-D numpy array and
            # append it as the last column to the preprocessed feature arrays.
            # Resulting shape: (n_samples, n_features + 1) where the final
            # column is the target (`math_score`). This is ready for model input.
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]  # last column = target
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]    # last column = target

            save_objec(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                object=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error("Error in initiate_data_transformation")
            raise CustomException(e, sys)

