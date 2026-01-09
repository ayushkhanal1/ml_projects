import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    """
    Complete training pipeline that orchestrates:
    1. Data Ingestion - Load and split data
    2. Data Transformation - Preprocess and transform data
    3. Model Training - Train multiple models and find the best one
    """
    
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
    
    def run_training_pipeline(self):
        """
        Execute the complete training pipeline from data ingestion to model training.
        
        Returns:
            tuple: (best_model_name, best_model_score)
        """
        try:
            logging.info("=" * 50)
            logging.info("TRAINING PIPELINE STARTED")
            logging.info("=" * 50)
            
            # Step 1: Data Ingestion
            logging.info("\nStep 1: Data Ingestion initiated...")
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"✓ Data Ingestion completed")
            logging.info(f"  - Train data: {train_data_path}")
            logging.info(f"  - Test data: {test_data_path}")
            
            # Step 2: Data Transformation
            logging.info("\nStep 2: Data Transformation initiated...")
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_data_path, 
                test_data_path
            )
            logging.info(f"✓ Data Transformation completed")
            logging.info(f"  - Train array shape: {train_arr.shape}")
            logging.info(f"  - Test array shape: {test_arr.shape}")
            logging.info(f"  - Preprocessor saved: {preprocessor_path}")
            
            # Step 3: Model Training
            logging.info("\nStep 3: Model Training initiated...")
            best_model_name, best_model_score = self.model_trainer.initiate_model_trainer(
                train_arr, 
                test_arr, 
                preprocessor_path
            )
            logging.info(f"✓ Model Training completed")
            logging.info(f"  - Best Model: {best_model_name}")
            logging.info(f"  - R2 Score: {best_model_score}")
            
            logging.info("\n" + "=" * 50)
            logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("=" * 50)
            
            return best_model_name, best_model_score
        
        except Exception as e:
            logging.error("Error occurred in Training Pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Execute the complete training pipeline
    pipeline = TrainingPipeline()
    best_model_name, best_model_score = pipeline.run_training_pipeline()
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Model Found: {best_model_name}")
    print(f"R2 Score: {best_model_score:.4f}")
    print("=" * 60)
