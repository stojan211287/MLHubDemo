import os

import keras
import tensorflow as tf

from keras.models import load_model

from mlhub.constants import MODEL_SAVE_DIR


class DeploymentError(Exception):
    pass


class DeployedModel:
    def __init__(self):

        self.save_path = os.path.join(".", MODEL_SAVE_DIR)

        self.code = None

        self.model = None
        self.tf_graph = None

    def load_model(self, model_code):

        self.code = model_code

        keras.backend.clear_session()
        self.tf_graph = tf.get_default_graph()

        try:
            model_path = os.path.join(self.save_path, "model_%s.hdf5") % (self.code,)

            if not os.path.exists(model_path):
                raise DeploymentError("Model model_%s does not exist!" % (model_code,))

            self.model = load_model(model_path, compile=False)

        except ValueError:
            raise DeploymentError("Error loading model model_%s" % (model_code,))

    def get_predictions(self, data):

        with self.tf_graph.as_default():
            prediction = self.model.predict(data)

        return prediction
