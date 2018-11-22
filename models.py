import importlib

import numpy as np
import pandas as pd

import torch
import tensorflow as tf

from models_errors import UserCodeExecutionError, InputDataError
from utils import exec_user_code


class TorchModel:

    def __init__(self, model_code):

        self.code = model_code

    def _parse_model_code(self, d_in):

        try:
            model_code = TorchModel._complete_model_code(user_code=self.code,
                                                         d_in=d_in)
            model_code_module = exec_user_code(code=model_code,
                                               module_name="custom_torch_model")
            # get model and loss_fn attributes
            model = getattr(model_code_module, "model")
            loss_fn = getattr(model_code_module, "loss_fn")

            return model, loss_fn

        except (ImportError, AttributeError, IndentationError):
            raise UserCodeExecutionError("There was an error in your model code.")

    @staticmethod
    def _complete_model_code(user_code, d_in):
        import_statements = "import torch;import torch.nn as nn;from torch.functional import F;"
        predefined_variables = "D_in=%d;D_out=1;" % (d_in,)

        return import_statements + predefined_variables + user_code

    def train(self, features, target_name, num_steps=100, learning_rate=1e-4):

        try:
            train_set, train_target, test_set, test_target = TorchModel._train_test_split(features=features,
                                                                                          target_name=target_name)

            train_set = TorchModel._convert_features(features=train_set)
            train_target = TorchModel._convert_target(target=train_target)

            test_set = TorchModel._convert_features(features=test_set)
            test_target = TorchModel._convert_target(target=test_target)

            model, loss_fn = self._parse_model_code(d_in=train_set.shape[1])

            for t in range(num_steps):

                preds = model(train_set)

                loss = loss_fn(preds, train_target)

                # TODO: REMOVE OR RETURN THIS
                print(t, loss.item())

                model.zero_grad()

                loss.backward()

                with torch.no_grad():
                    for param in model.parameters():
                        param -= learning_rate * param.grad

            return model

        except ValueError as error:
            raise InputDataError("Error while converting input data: %s" % (str(error), ))

        except RuntimeError as error:
            raise UserCodeExecutionError("Error while training model: %s" %(str(error), ))

    @staticmethod
    def _convert_features(features):

        if isinstance(features, pd.DataFrame):
            return torch.tensor(features.values)
        else:
            raise ValueError("Features must be a pandas DataFrame!")

    @staticmethod
    def _convert_target(target):

        if isinstance(target, pd.Series):
            target = torch.tensor(target.values)
        else:
            raise ValueError("Target must be a pandas Series!")

        target = target.type(torch.FloatTensor)

        return target.unsqueeze_(1)

    @staticmethod
    def _train_test_split(features: pd.DataFrame, target_name: str, test_set_size: float = 0.1):

        train_indices = np.random.choice(features.index.values,
                                         size=int((1. - test_set_size) * features.shape[0]))

        test_indices = np.setdiff1d(features.index.values,
                                    train_indices,
                                    assume_unique=True)

        train_features, test_features = features.iloc[train_indices, :], features.iloc[test_indices, :]

        train_target = train_features[target_name]
        test_target = test_features[target_name]

        # SEPARATE TARGET COLUMN AND REST OF FEATURES
        training_set = train_features.drop(target_name, axis=1)
        test_set = test_features.drop(target_name, axis=1)

        return training_set, train_target, test_set, test_target


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







