import hashlib


class User:

    def __init__(self, datasets):

        self.available_datasets = datasets

        self.loaded_data = None
        self.loaded_data_name = None

        self.feature_code = None
        self.model_code = None

        self.committed_features = None

    def commit_features(self, feature_preview):

        if self.committed_features is None:
            self.committed_features = dict()

        feature_hash = hashlib.md5(feature_preview.encode()).hexdigest()
        user_code_list = self.feature_code.split(";")[:-1]

        self.committed_features[feature_hash] = {"code": user_code_list,
                                                 "raw_data": self.loaded_data_name}

        return feature_hash

    def add_dataset(self, url, dataset_name, target=None):

        self.available_datasets[dataset_name] = {"URL": url,
                                                 "target": target}

    def get_feature_def_list(self, commit_hash):
        return self.committed_features[commit_hash]["code"]

    def get_commit_data_name(self, commit_hash):
        return self.committed_features[commit_hash]["raw_data"]
