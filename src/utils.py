import os
import sys
import joblib
import numpy as np
import tensorflow as tf
from src.exception import CustomException
import faiss

def save_keras(model, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        model.save(file_path)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_keras(file_path):
    try:
        return tf.keras.models.load_model(file_path)
    except Exception as e:
        raise CustomException(e,sys)
    
def save_npy(data, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        np.save(file_path, data)
    except Exception as e:
        raise CustomException(e, sys)

def load_npy(file_path):
    try:
        return np.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)

def save_joblib(model, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(model, file_path)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_joblib(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)
    
def save_faiss(index, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        faiss.write_index(index, file_path)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_faiss(file_path):
    try:
        return faiss.read_index(file_path)
    except Exception as e:
        raise CustomException(e, sys)