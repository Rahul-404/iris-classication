import pickle
import os
import sys

import numpy as np
import pandas as pd

import dill

from src.exception import CustomException

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