import numpy as np
import pandas as pd

import torch

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








