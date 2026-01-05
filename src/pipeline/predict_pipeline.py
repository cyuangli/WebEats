import sys
import os
import tensorflow as tf
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import ConfigurationManager
from src.utils import load_keras, load_npy, load_faiss, load_joblib

class PredictionPipeline():

    def __init__(self):
        config_manager = ConfigurationManager()
        self.model_paths = config_manager.get_model_training_config().save_path
    
    def load_and_preprocess_image(self, path, img_size=(224,224)):
        img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255.0
        return np.expand_dims(img, axis=0)
    
    def initiate_pipeline(self, data, k):
        logging.info("Initiating the prediction pipeline.")

        try:
            logging.info("Loading models and data.")

            embedding_model = load_keras(os.path.join(self.model_paths, "embedding_model.keras"))
            pca = load_joblib(os.path.join(self.model_paths, "pca.joblib"))
            index = load_faiss(os.path.join(self.model_paths, "recipes.faiss"))
            image_paths = load_npy(os.path.join(self.model_paths, "image_paths.npy"))
            image_paths = image_paths.astype(str)

            logging.info("Transforming data.")

            img = self.load_and_preprocess_image(data)

            embedding = embedding_model.predict(img)
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

            embedding_pca = pca.transform(embedding)

            _, indices = index.search(embedding_pca, k)
            results = indices[0]

            images = []
            for idx in results:
                images.append(image_paths[idx])

            return images

        except Exception as e:
            raise CustomException(e, sys)