import os

import numpy as np

from data import DataLoader
from features import FeatureDef, FeatureParser
from models import TFModel
from utils import ensure_local_dir


def main():

    data_loader = DataLoader(local_data_dir=os.environ["MFH_LOCAL_DATA_DIR"])
    data = data_loader.load_data(data_path=URL)

    features = [
        FeatureDef(name="log1pOf1",
                   kind="numeric",
                   recipe={"generators": [1],
                                   "function": np.log1p
                           }),
        FeatureDef(name="sumOfLog1pOf1And2",
                   kind="numeric",
                   recipe={"generators": [1, 2],
                                   "function": lambda x, y: np.log1p(x) + np.log1p(y)}),
        FeatureDef(name="log1pOfSumOf1And2",
                   kind="numeric",
                   recipe={"generators": [1, 2],
                                   "function": lambda x, y: np.log1p(np.array(x) + np.array(y))}),
        FeatureDef(name="3rdFeature",
                   kind="numeric",
                   recipe={"generators": [3],
                                   "function": lambda x: np.array(x)}),
    ]

    parser = FeatureParser(features=features)

    parsed_data_df = parser.parse_to_df(data)
    target = data.loc[:, 0].apply(lambda x: x - 1)

    model_params = {"hidden_units": [3, 5, 3],
                    "n_classes": 3}

    model = TFModel(tf_estimator_class="DNNClassifier",
                    model_parameters=model_params,
                    feature_parser=parser,
                    model_export_directory=os.environ["MFH_LOCAL_MODEL_DIR"])

    model.train_and_eval(features=parsed_data_df,
                         target=target,
                         num_steps=100)

    model.deploy()


if __name__ == "__main__":

    os.environ["MFH_LOCAL_DATA_DIR"] = os.path.join(".", "data")
    os.environ["MFH_LOCAL_MODEL_DIR"] = os.path.join(".", "models")

    ensure_local_dir(os.environ["MFH_LOCAL_DATA_DIR"])
    ensure_local_dir(os.environ["MFH_LOCAL_MODEL_DIR"])

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

    main()
