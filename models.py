import importlib

import pandas as pd
import tensorflow as tf


class TFModel:

    def __init__(self, tf_estimator_class, model_parameters, feature_parser, model_export_directory):

        self.tf_class_name = tf_estimator_class
        self.model_params = model_parameters
        self.parser = feature_parser
        self.export_dir = model_export_directory

        self.model_instance = None

        try:
            self.model_class = getattr(importlib.import_module("tensorflow.estimator"), self.tf_class_name)
        except ImportError:
            raise Exception("Cannot import tensorflow.estimator")
        except AttributeError:
            raise Exception("There does not seem to be a %s class in tensorflow.estimator" % (self.tf_class_name, ))

    def train(self, features, target, num_steps):

        try:
            assert isinstance(features, pd.DataFrame)
            assert isinstance(target, pd.Series)

        except AssertionError:
            raise ValueError("The TFModel.train has the signature features: pd.DataFrame, target: pd.Series")

        training_input_fn = tf.estimator.inputs.pandas_input_fn(x=features,
                                                                y=target,
                                                                batch_size=features.shape[0],
                                                                num_epochs=num_steps,
                                                                shuffle=False)
        instance_model_params = self.model_params.copy()
        instance_model_params["feature_columns"] = self.parser.get_tf_feature_columns(data=features)

        self.model_instance = self.model_class(**instance_model_params)

        self.model_instance.train(input_fn=training_input_fn,
                                  steps=num_steps)

    def deploy(self):

        production_features = {}

        for feature_definition in self.parser.features:

            tf_feature_type = tf.float32 if feature_definition.kind == "numeric" else tf.string
            production_features[feature_definition.name] = tf.FixedLenFeature(shape=[1],
                                                                              dtype=tf_feature_type)

        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec=production_features,
                                                                                       default_batch_size=None)

        self.model_instance.export_savedmodel(export_dir_base=self.export_dir,
                                              serving_input_receiver_fn=serving_input_fn)







