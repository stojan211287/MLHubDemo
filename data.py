import os
import shutil
import pandas as pd

import urllib.request
from urllib.error import URLError

from utils import ensure_local_dir


class DataNotFoundLocally(Exception):
    pass


class DataLoader:

    def __init__(self, local_data_dir):

        self.local_data_dir = local_data_dir
        ensure_local_dir(self.local_data_dir)

    def load_data(self, data_path):

        try:
            assert isinstance(data_path, str)

        except AssertionError:
            raise ValueError("data_path must be a string URL or absolute path!")

        if data_path.startswith("http"):
            downloaded_file = self._download_and_cache_data(data_url=data_path)

            return pd.read_csv(downloaded_file, header=None)
        else:
            try:
                return pd.read_csv(data_path)
            except IOError:
                raise DataNotFoundLocally

    def _download_and_cache_data(self, data_url):

        dataset_name = data_url.split("/")[-1]

        save_file_name = os.path.join(self.local_data_dir, dataset_name)

        # CHECK IF DATA HAS ALREADY BEEN DOWNLOADED
        if os.path.exists(save_file_name):
            print("Data at %s already found locally. Loading..." % (data_url, ))
        else:
            print("Data at %s not found locally. Downloading..." % (data_url, ))
            try:
                with urllib.request.urlopen(data_url) as response, \
                open(save_file_name, "wb") as out_file:

                    shutil.copyfileobj(response, out_file)

            except URLError:
                raise ValueError("File %s not found at %s" % (save_file_name, data_url))

        return save_file_name


if __name__ == "__main__":

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

    data_loader = DataLoader(local_data_dir="./data")
    data = data_loader.load_data(data_path=URL)

    print(data.head())