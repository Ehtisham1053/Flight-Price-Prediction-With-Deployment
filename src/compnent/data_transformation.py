import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from src.exception import Custom_exception_handling
from src.logger import logging
from sklearn.pipeline import Pipeline
from src.utils import save_object
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    label_encoders_file_path: str = os.path.join("artifacts", "label_encoders.pkl")  # Add this line


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.label_encoders = {}

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Date and Time feature extraction
            df['Date'] = df['Date_of_Journey'].str.split('/').str[0].astype(int)
            df['Month'] = df['Date_of_Journey'].str.split('/').str[1].astype(int)
            df['Year'] = df['Date_of_Journey'].str.split('/').str[2].astype(int)
            df.drop('Date_of_Journey', axis=1, inplace=True)

            df['Arrival_Time'] = df['Arrival_Time'].str.split(' ').str[0]
            df['Arrival_Hour'] = df['Arrival_Time'].str.split(':').str[0].astype(int)
            df['Arrival_Min'] = df['Arrival_Time'].str.split(':').str[1].astype(int)
            df.drop('Arrival_Time', axis=1, inplace=True)

            df['Dept_hour'] = df['Dep_Time'].str.split(':').str[0].astype(int)
            df['Dept_min'] = df['Dep_Time'].str.split(':').str[1].astype(int)
            df['Dept_min'] = df['Dept_min'].fillna(0)
            df.drop('Dep_Time', axis=1, inplace=True)

            # Mapping 'Total_Stops'
            df['Total_Stops'] = df['Total_Stops'].map({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})
            df.drop('Route', axis=1, inplace=True)

            # Duration conversion to minutes
            df['Duration_hour'] = df['Duration'].str.split(' ').str[0].str.split('h').str[0].astype(int)
            df['Duration_min'] = df['Duration'].str.split(' ').str[1].str.split('m').str[0].fillna(0).astype(int)
            df['Duration_in_min'] = df['Duration_hour'] * 60 + df['Duration_min']
            df.drop(['Duration', 'Duration_hour', 'Duration_min'], axis=1, inplace=True)

            # Categorical encoding with LabelEncoder
            categorical_features = ['Airline', 'Source', 'Destination', 'Additional_Info']
            for feature in categorical_features:
                if feature not in self.label_encoders:
                    le = LabelEncoder()
                    df[feature] = le.fit_transform(df[feature])
                    self.label_encoders[feature] = le
                else:
                    le = self.label_encoders[feature]
                    df[feature] = le.transform(df[feature])

            # Drop duplicates after transformation
            df.drop_duplicates(inplace=True)
            
            return df

        except Exception as e:
            raise Custom_exception_handling(e, sys)

    def get_data_transformation(self):
        try:
            logging.info("Creating preprocessing pipeline")

            # Define feature sets
            numerical_features = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Additional_Info',
                                  'Date', 'Month', 'Year', 'Arrival_Hour', 'Arrival_Min',
                                  'Dept_hour', 'Dept_min', 'Duration_in_min']

            transformers = []

            # Numerical pipeline
            if numerical_features:
                transformers.append(
                    ("num_pipeline", Pipeline(
                        steps=[
                            ("standard_scaler", StandardScaler())
                        ]
                    ), numerical_features)
                )

            # Combining pipelines
            preprocessor = ColumnTransformer(transformers=transformers)

            logging.info("Preprocessing pipeline created")
            return preprocessor

        except Exception as e:
            raise Custom_exception_handling(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Loading training and testing data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Apply preprocessing
            logging.info("Preprocessing training data")
            train_df = self.preprocess_data(train_df)
            logging.info("Preprocessing testing data")
            test_df = self.preprocess_data(test_df)

            # Target column
            target_column = "Price"
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing pipeline to training and testing data")
            preprocessor_obj = self.get_data_transformation()

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing pipeline object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Saving label encoders object")
            with open(self.data_transformation_config.label_encoders_file_path, "wb") as f:
                pickle.dump(self.label_encoders, f)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise Custom_exception_handling(e, sys)

