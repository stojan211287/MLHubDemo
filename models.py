import importlib

import numpy as np
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
            assert isinstance(target, str)

        except AssertionError:
            raise ValueError("The TFModel.train has the signature features: pd.DataFrame, target: String")

        train_features, test_features = TFModel._train_test_split(features=features)

        train_target = train_features[target]
        test_target = test_features[target]

        # SEPARATE TARGET COLUMN AND REST OF FEATURES
        training_set = train_features.drop(target, axis=1)
        test_set = test_features.drop(target, axis=1)

        training_input_fn = tf.estimator.inputs.pandas_input_fn(x=training_set,
                                                                y=train_target,
                                                                batch_size=training_set.shape[0],
                                                                num_epochs=num_steps,
                                                                shuffle=False)

        test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_set,
                                                            y=test_target,
                                                            batch_size=test_set.shape[0],
                                                            num_epochs=1,
                                                            shuffle=False)

        instance_model_params = self.model_params.copy()
        instance_model_params["feature_columns"] = self.parser.get_tf_feature_columns(data=features)

        self.model_instance = self.model_class(**instance_model_params)

        train_spec = tf.estimator.TrainSpec(input_fn=training_input_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn)

        evaluations, _ = tf.estimator.train_and_evaluate(self.model_instance, train_spec, eval_spec)

        class_predictions = []

        for prediction_wrapper in self.model_instance.predict(input_fn=test_input_fn):
            class_predictions.append(prediction_wrapper["class_ids"][0])

        test_features.loc[:, self.tf_class_name+"_predictions"] = class_predictions

        return evaluations, test_features

    @staticmethod
    def _train_test_split(features, test_set_size=0.1):

        train_indices = np.random.choice(features.index.values,
                                         size=int((1.-test_set_size)*features.shape[0]))

        test_indices = np.setdiff1d(features.index.values,
                                    train_indices,
                                    assume_unique=True)

        return features.iloc[train_indices, :], features.iloc[test_indices, :]

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







