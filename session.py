import hashlib


class UserSession:

    def __init__(self, datasets):

        self.logged_in = False

        self.available_datasets = datasets
        self.available_models = dict()

        self.loaded_data = None
        self.loaded_data_name = None

        self.feature_code = None
        self.model_code = None

        self.committed_features = None
        self.latest_feature_commit = None

        self.latest_predictions = None

        self.latest_trained_model = None

    def commit_features(self, feature_preview, all_features):

        if self.committed_features is None:
            self.committed_features = dict()

        feature_hash = hashlib.md5(feature_preview.encode()).hexdigest()
        user_code_list = self.feature_code.split(";")[:-1]

        self.committed_features[feature_hash] = {"code": user_code_list,
                                                 "raw_data": self.loaded_data_name,
                                                 "all_features": all_features}
        self.latest_feature_commit = feature_hash

    def load_data(self):
        pass

    def commit_model(self):
        pass

    def add_dataset(self, url, dataset_name):

        self.available_datasets[dataset_name] = url

    def get_feature_def_list(self, commit_hash):
        return self.committed_features[commit_hash]["code"]

    def get_commit_data_name(self, commit_hash):
        return self.committed_features[commit_hash]["raw_data"]
