import os
import sys
import types

from data import DataLoader
from features import FeatureParser

default_feature_code = \
"""FeatureDef(
        name="log1pOfFixedAcidity",
        kind="numeric",
        recipe={
                "generators": ["fixed acidity"],
                "function": np.log1p
                }
);
FeatureDef(
        name="sumOfLog1pOfpHAndDensity",
        kind="numeric",
        recipe={
                "generators": ["pH", "density"],
                "function": lambda x, y: np.log1p(x)+np.log1p(y)
                }
);
FeatureDef(
        name="log1pOfSumOfpHAndDensity",
        kind="numeric",
        recipe={
                "generators": ["pH", "density"],
                "function": lambda x, y: np.log1p(x+y)
                }
);
"""


model_param_lookup = {
                        "DNNClassifier": {
                                        "hidden_units": [3, 5, 3]},
                        "BoostedTreesClassifier": {"n_trees": 100,
                                                   "max_depth": 6,
                                                   "learning_rate": 0.1,
                                                   "n_batches_per_layer": 1},
                        "LinearClassifier": {}}


def exec_user_code(code, module_name="feature_trans"):

    user_module_path = os.path.join(".", ".user_module.py")

    mod = types.ModuleType(module_name)
    mod.__file__ = user_module_path
    sys.modules[module_name] = mod

    exec(code, mod.__dict__)

    return mod


def complete_user_code(user_code):

    import_statements = """from features import FeatureDef;import numpy as np;"""

    feature_def_elements = user_code.split(";")[:-1]  # ELIMINATE POSSIBLE STUFF AFTER THE LAST SEMICOLON
    feature_list_code = "features = ["+",".join(feature_def_elements)+"]"

    return import_statements+feature_list_code


def get_data(url):
    data_loader = DataLoader(local_data_dir="./data")
    return data_loader.load_data(data_path=url)


def encode_target(raw_target):

    encoded_target = raw_target.astype(int)

    unique_target_values = encoded_target.unique()
    n_classes = len(unique_target_values)

    target_value_lookup = {true_target_value: target_value_code
                           for target_value_code, true_target_value in enumerate(sorted(unique_target_values))}

    for i, value in enumerate(encoded_target):
        encoded_target[i] = target_value_lookup[value]

    return encoded_target, n_classes


def construct_parser(code_raw_string):

    complete_code = complete_user_code(user_code=code_raw_string)
    module_with_user_code = exec_user_code(code=complete_code)
    feature_list = getattr(module_with_user_code, "features")

    return FeatureParser(features=feature_list)
