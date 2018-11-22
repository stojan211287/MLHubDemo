BASE_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
NO_OF_ROWS_TO_SHOW = 15
TRACEBACK_LIMIT = 2

DATASETS = {
    "RedWineQuality": {"URL": BASE_DATA_URL+"wine-quality/winequality-red.csv",
                       "target": "quality"},
    "WhiteWineQuality": {"URL": BASE_DATA_URL+"wine-quality/winequality-white.csv",
                         "target": "quality"},
    "BreastCancerWisconsinDataset": {"URL":BASE_DATA_URL+"breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                                     "target": 10},
    "ErrorDataset": {"URL": BASE_DATA_URL+"this-is-not-a-dataset.fsv",
                     "target": "there_is_no_target"}
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
H_1 = 3
H_2 = 5
H_3 = 3

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H_1),
    torch.nn.ReLU(),
    torch.nn.Linear(H_1, H_2),
    torch.nn.ReLU(),
    torch.nn.Linear(H_2, H_3),
    torch.nn.ReLU(),
    torch.nn.Linear(H_3, D_out),
)

loss_fn = torch.nn.MSELoss(reduction='sum')
"""

model_param_lookup = {
                        "DNNClassifier": {
                                        "hidden_units": [3, 5, 3]},
                        "BoostedTreesClassifier": {"n_trees": 100,
                                                   "max_depth": 6,
                                                   "learning_rate": 0.1,
                                                   "n_batches_per_layer": 1},
                        "LinearClassifier": {}}