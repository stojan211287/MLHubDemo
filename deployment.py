import os
import threading
import tensorflow as tf

from keras.models import load_model
from flask import Flask, request

from constants import MODEL_SAVE_DIR


class DeployedModel(threading.Thread):

    def __init__(self, model_code):

        threading.Thread.__init__(self)

        self.code = model_code
        self.app = Flask("running_model")

        save_path = os.path.join(".", MODEL_SAVE_DIR)

        model = load_model(os.path.join(save_path, "model_%s.hdf5" % (model_code, )),
                           compile=False)

        graph = tf.get_default_graph()

        @self.app.route(rule="/predict", methods=["POST"])
        def predict():
            with graph.as_default():

                post_data = request.json
                prediction = model.predict(post_data)

            return prediction

    def run(self):

        self.app.run(host="0.0.0.0",
                     port=5000,
                     debug=False)
