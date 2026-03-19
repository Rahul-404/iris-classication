import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """ 
        This function is responsible for data transformation
        """
        try:
            numerical_columns = [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)"
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Numeircal columns standard scaling completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, numerical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Obtained preprocessing object")

            target_column_name = "target"
            numerical_columns = [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)"
            ]

            input_fetaure_train_df = train_df.drop(columns=[target_column_name, "species"], axis=1)
            target_fetaure_train_df = train_df[target_column_name]

            input_fetaure_test_df = test_df.drop(columns=[target_column_name, "species"], axis=1)
            target_fetaure_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocesing object on training dataframe and testing dataframe"
            )

            input_fetaure_train_arr = preprocessing_obj.fit_transform(input_fetaure_train_df)
            input_fetaure_test_arr = preprocessing_obj.transform(input_fetaure_test_df)

            train_arr = np.c_[input_fetaure_train_arr, np.array(target_fetaure_train_df)]
            test_arr = np.c_[input_fetaure_test_arr, np.array(target_fetaure_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            CustomException(e, sys)