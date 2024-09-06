import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.exception import Custom_exception_handling
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, feature_columns, target_column):
        try:
            logging.info("Split training and test input data")

            # Identify the index of the target column 'Price'
            target_index = feature_columns.index(target_column)

            # Split the data into features and target
            X_train = train_array[:, [i for i in range(train_array.shape[1]) if i != target_index]]
            y_train = train_array[:, target_index]
            X_test = test_array[:, [i for i in range(test_array.shape[1]) if i != target_index]]
            y_test = test_array[:, target_index]

            # Initialize and train the RandomForestRegressor model
            model = RandomForestRegressor(
                max_depth=30, 
                min_samples_leaf=1, 
                min_samples_split=2, 
                n_estimators=300
            )
            model.fit(X_train, y_train)
            
            # Predict on the test set
            y_test_pred = model.predict(X_test)

            # Calculate R-squared score
            test_model_score = r2_score(y_test, y_test_pred)

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            # Convert R-squared score to percentage
            accuracy = test_model_score * 100
            logging.info(f"Model accuracy: {accuracy:.2f}%")

            return accuracy

        except Exception as e:
            raise Custom_exception_handling(e, sys)

