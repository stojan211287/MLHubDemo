import numpy as np
import pandas as pd
import tensorflow as tf

from collections import namedtuple
from operator import itemgetter


class FeatureDef:

        def __init__(self, name, kind, recipe=None):

            self.name = name
            self.kind = kind
            self.recipe = recipe

            self.wrapper = namedtuple("Feature", ["name", "kind", "values"])

        def _ensure_type(self, feature_values):
            if self.kind == "numeric":
                return np.array(feature_values, dtype=np.float32)
            elif self.kind == "categorical":
                return np.array(feature_values, dtype=np.unicode_)
            else:
                raise ValueError("Unknown feature kind %s" % (self.kind, ))

        @staticmethod
        def _ensure_numpy(feature_values):
            if not isinstance(feature_values, np.ndarray):
                return np.array(feature_values)
            else:
                return feature_values

        def parse(self, data):

            if isinstance(data, pd.DataFrame):
                    data_dict = data.to_dict(orient="list")
            elif isinstance(data, dict):
                    data_dict = data
            else:
                raise ValueError("Variable data can be either a Pandas Dataframe or a dictionary!")

            for feature_name, feature_values in data_dict.items():
                data_dict[feature_name] = FeatureDef._ensure_numpy(feature_values)

            try:
                feature_values = data_dict[self.name]
            except KeyError:
                if self.recipe is None:
                    raise ValueError("You most provide a recipe for feature transformation!")
                else:
                    feature_generators = itemgetter(*self.recipe["generators"])(data_dict)

                    if isinstance(feature_generators, tuple):
                        feature_values = self.recipe["function"](*feature_generators)
                    else:
                        feature_values = self.recipe["function"](feature_generators)

            return self.wrapper(name=self.name, kind=self.kind, values=self._ensure_type(feature_values))


class FeatureParser:

        def __init__(self, features):

            self.features = []

            for feature in features:
                try:
                    assert isinstance(feature, FeatureDef)
                    self.features.append(feature)
                except AssertionError:
                    raise ValueError("You most provide a list of FeatureDefs to FeatureParser!")

        def parse(self, data):

            parsed_feature_dict = {}

            for feature in self.features:
                parsed_feature_dict[feature.name] = feature.parse(data=data)

            return parsed_feature_dict

        def parse_to_df(self, data):

            parsed_data_dict = self.parse(data)

            for feature_name, parsed_feature in parsed_data_dict.items():
                parsed_data_dict[feature_name] = parsed_feature.values

            return pd.DataFrame.from_dict(parsed_data_dict)

        def get_tf_feature_columns(self, data):

            feature_columns = []

            for feature in self.features:

                feature_key = str(feature.name)

                if feature.kind == "numeric":
                    feature_columns.append(tf.feature_column.numeric_column(key=feature_key,
                                                                            dtype=tf.float32))
                elif feature.kind == "categorical":
                    feature_vocab = np.unique(feature.parse(data))
                    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(key=feature_key,
                                                                                                     dtype=tf.string,
                                                                                                     vocabulary_list=
                                                                                                     feature_vocab))
                else:
                    raise ValueError("Feature kinds can only be 'numeric' or 'categorical', not %s" % (feature.kind, ))

            return feature_columns


if __name__ == "__main__":

    import numpy as np
    from data import DataLoader

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

    data_loader = DataLoader(local_data_dir="./data")
    data = data_loader.load_data(data_path=URL)

    print(data.head())
    print(data.columns)

    features = [
                FeatureDef(name="log1pOf1",
                           kind="numeric",
                           recipe={
                                    "generators": [1],
                                    "function": np.log1p
                                   }),
                FeatureDef(name="sumOfLog1pOf1And2",
                           kind="numeric",
                           recipe={
                                    "generators": [1, 2],
                                    "function": lambda x, y: np.log1p(x)+np.log1p(y)}),
                FeatureDef(name="log1pOfSumOf1And2",
                           kind="numeric",
                           recipe={
                                    "generators": [1, 2],
                                    "function": lambda x, y: np.log1p(np.array(x)+np.array(y))}),
                FeatureDef(name=3,
                           kind="numeric"),
                FeatureDef(name="3rdFeature",
                           kind="numeric",
                           recipe={
                                    "generators": [3],
                                    "function": lambda x: np.array(x)}),
                ]

    parser = FeatureParser(features=features)
    parsed_data_df = parser.parse_to_df(data)

    print(parsed_data_df.head())
