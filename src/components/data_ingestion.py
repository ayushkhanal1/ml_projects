import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:                  #giving input to the data ingestion component
    train_data_path: str = os.path.join('artifacts', 'train.csv')  # where the training split will be saved
    test_data_path: str = os.path.join('artifacts', 'test.csv')    # where the test split will be saved
    raw_data_path: str = os.path.join('artifacts', 'data.csv')     # a copy of the original/raw dataset


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            # Read the raw CSV. Update this path if your data is located elsewhere.
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Dataset read as dataframe")

            # Ensure the artifacts directory exists (dirname of train path -> 'artifacts')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save a copy of the raw data before any processing/splitting
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            # Split the dataframe into train and test. `random_state` keeps the split reproducible.
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Persist the train and test splits to the configured locations
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            # Return the file paths for downstream steps to use
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            # Wrap the caught exception in `CustomException` to include traceback details
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()   
    print(f"Train data path: {train_data}\nTest data path: {test_data}")
               