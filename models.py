import os
import keras

import numpy as np
import pandas as pd

from keras import metrics
from keras.optimizers import SGD

from models_errors import UserCodeExecutionError, InputDataError, DeploymentError
from utils import exec_user_code, encode_target
from constants import MODEL_SAVE_DIR


class KerasModel:

    def __init__(self, model_code):

        self.code = model_code
        self.model_instance = None

    def _parse_model_code(self, d_in, d_out):

        try:
            model_code = KerasModel._complete_model_code(user_code=self.code,
                                                         d_in=d_in,
                                                         d_out=d_out)

            print(model_code)

            model_code_module = exec_user_code(code=model_code,
                                               module_name="custom_keras_model_%s" % (model_code, ))

            # GET MODEL ATTRIBUTE FROM USER CODE
            model = getattr(model_code_module, "model")

            return model

        except (ImportError, AttributeError, IndentationError) as error:
            raise UserCodeExecutionError("There was an error in your model code: %s" % (str(error), ))

    @staticmethod
    def _complete_model_code(user_code, d_in, d_out):
        import_statements = ";".join(["import keras",
                                      "from keras.models import Sequential",
                                      "from keras.layers import Dense, Dropout, Activation"])+";"

        predefined_variables = "D_in=%d;D_out=%d;" % (d_in, d_out)

        return import_statements + predefined_variables + user_code

    def train_and_eval(self, features, target_name, epochs=100, learning_rate=1e-2):

        try:
            num_classes = len(np.unique(features[target_name].values))

            # ENCODE TARGET
            features[target_name] = encode_target(raw_target=features[target_name])

            train_set, train_target, test_set, test_target = KerasModel._train_test_split(features=features,
                                                                                          target_name=target_name)

            train_set_np = KerasModel._convert_features(features=train_set)
            train_target_np = KerasModel._convert_target(target=train_target,
                                                         num_classes=num_classes)

            test_set_np = KerasModel._convert_features(features=test_set)
            test_target_np = KerasModel._convert_target(target=test_target,
                                                        num_classes=num_classes)

            train_batch = int(train_set_np.shape[0] * 0.25)

            # CLEAR CACHED Tensorflow DATA IN ORDER TO BE ABLE TO ITERATE OVER MODELS
            keras.backend.clear_session()

            model = self._parse_model_code(d_in=train_set_np.shape[1],
                                           d_out=num_classes)

            sgd = SGD(lr=learning_rate,
                      decay=1e-6,
                      momentum=0.9,
                      nesterov=True)

            model.compile(loss="categorical_crossentropy",
                          optimizer=sgd,
                          metrics=["accuracy", metrics.categorical_accuracy])

            model.fit(train_set_np,
                      train_target_np,
                      epochs=epochs,
                      batch_size=train_batch)

            # SAVE LAST TRAINED MODEL
            self.model_instance = model

            metric_names = ["target_name", "categorical_crossentropy", "categorical_accuracy"]
            eval_metrics = model.evaluate(test_set_np,
                                          test_target_np,
                                          batch_size=test_set_np.shape[0])
            eval_metrics = [target_name] + eval_metrics
            eval_dict = dict(zip(metric_names, eval_metrics))

            predicted_classes = model.predict(x=test_set_np,
                                              batch_size=test_set_np.shape[0])

            test_set_with_preds = test_set
            test_set_with_preds["target"] = test_target
            test_set_with_preds["target_predictions"] = predicted_classes.argmax(axis=-1)

            return eval_dict, test_set_with_preds

        except ValueError as error:
            raise InputDataError("Error while converting input data: %s" % (str(error), ))

        except RuntimeError as error:
            raise UserCodeExecutionError("Error while training model: %s" %(str(error), ))

    def deploy(self, model_id):

        if self.model_instance is not None:

            save_dir = os.path.join(".", MODEL_SAVE_DIR)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            self.model_instance.save(os.path.join(save_dir, "model_%s.hdf5" % (model_id, )))

        else:
            raise DeploymentError("Error deploying model: Model has not been trained!")

    @staticmethod
    def _convert_features(features):
        return features.values

    @staticmethod
    def _convert_target(target, num_classes):
        return keras.utils.to_categorical(target.values, num_classes=num_classes)

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




