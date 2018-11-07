import os
import sys
import types

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
                "function": lambda x, y: np.log1p(np.array(x)+np.array(y))
                }
);
"""


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


def ensure_local_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        pass
