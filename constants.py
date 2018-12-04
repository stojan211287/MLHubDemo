BASE_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
NO_OF_ROWS_TO_SHOW = 15
TRACEBACK_LIMIT = 2

MODEL_SAVE_DIR = "saved_models"

ADMIN = "admin"
SUPER_SAFE_ADMIN_PASSWORD = "admin"

DATASETS = {
    "RedWineQuality": BASE_DATA_URL+"wine-quality/winequality-red.csv",
    "WhiteWineQuality": BASE_DATA_URL+"wine-quality/winequality-white.csv",
    "BreastCancerWisconsinDataset": BASE_DATA_URL+"breast-cancer-wisconsin/breast-cancer-wisconsin.data",
    "ErrorDataset": BASE_DATA_URL+"this-is-not-a-dataset.fsv",
}

DATA_ERRORS = (NameError, SyntaxError, AttributeError, KeyError, ValueError, TypeError)
TRAINING_ERRORS = (NameError, SyntaxError, AttributeError, KeyError, ValueError, TypeError)


default_feature_code = \
"""
FeatureDef(
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

default_model_code = \
"""
# This is a pre-defined multi-class MLP model

# Notice that it is using values D_in and D_out
# These are variables, pre-defined by MLHub

# D_in = number of features in chosen feature commit 
# D_out = number of unique values in the chosen target feature

H_1 = 20
H_2 = 30
H_3 = 20

model = Sequential()

model.add(Dense(H_1, activation='relu', input_dim=D_in))
model.add(Dropout(0.5))

model.add(Dense(H_2, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(H_3, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(D_out, activation='softmax'))
"""

model_param_lookup = {
                        "DNNClassifier": {
                                        "hidden_units": [10, 12, 10]},
                        "BoostedTreesClassifier": {"n_trees": 100,
                                                   "max_depth": 6,
                                                   "learning_rate": 0.1,
                                                   "n_batches_per_layer": 1},
                        "LinearClassifier": {}}
