import os
import sys

import numpy as np 
import pandas as pd
# import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import Custom_exception_handling

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise Custom_exception_handling(e, sys)
    
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def evaluate_model(X_train, y_train, X_test, y_test):
    try:
        # Directly create the model with specified parameters
        model = RandomForestRegressor(
            max_depth=30, 
            min_samples_leaf=1, 
            min_samples_split=2, 
            n_estimators=300
        )
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)

        test_model_score = r2_score(y_test, y_test_pred)

        return test_model_score

    except Exception as e:
        raise Custom_exception_handling(e, sys)



    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise Custom_exception_handling(e, sys)
