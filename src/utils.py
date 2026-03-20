import pickle
import os
import sys

import numpy as np
import pandas as pd

import dill

from src.exception import CustomException

from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import GridSearchCV

def save_object(file_path:str, obj):
    try:
        # Correctly get only the folder path
        dir_path = os.path.dirname(file_path)
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params, cv=3, n_jobs=3, verbose=False):
    try:
        report = {}

        for i in range(len(list(models.keys()))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv=cv, n_jobs=n_jobs, verbose=verbose, refit=True)
            gs.fit(X_train, y_train) # Train model

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = f1_score(y_train, y_train_pred, average='weighted')
            test_model_score = f1_score(y_test, y_test_pred, average='weighted')

            report[list(models.keys())[i]] = test_model_score
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)