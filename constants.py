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