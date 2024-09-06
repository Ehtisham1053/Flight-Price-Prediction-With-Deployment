import os 
import sys 
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import Custom_exception_handling
from src.logger import logging
from dataclasses import dataclass


from src.compnent.data_transformation import DataTransformation
from src.compnent.data_transformation import DataTransformationConfig

from src.compnent.model_training import ModelTrainer
from src.compnent.model_training import ModelTrainerConfig



# path for saving  the test , train and raw data 
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # this first line we can change whenever we are reading the data from any source 
            df=pd.read_excel('data_set\data\data.xlsx')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            # now train and test data split is completed
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                # this return is for the data transformation part

            )
        except Exception as e:
            raise Custom_exception_handling(e,sys)
        


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    # data transformation and creation on pkl file
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    # # model training and returing of the accuracy
    modeltrainer = ModelTrainer()
    train_features = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Additional_Info',
                      'Date', 'Month', 'Year', 'Arrival_Hour', 'Arrival_Min',
                      'Dept_hour', 'Dept_min', 'Duration_in_min', 'Price']
    target_column = 'Price'
    accuracy = modeltrainer.initiate_model_trainer(train_arr, test_arr, train_features, target_column)
    print(f"Model accuracy: {accuracy:.2f}%")



