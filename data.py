import os
import shutil
import pandas as pd

import urllib.request
from urllib.error import URLError


class DataLoadingError(Exception):
    pass


class DataNotFoundLocally(DataLoadingError):
    pass


class DataNotFoundRemotly(DataLoadingError):
    pass


class DatasetFormatNotSupported(DataLoadingError):
    pass


class MalformedDataUrl(DataLoadingError):
    pass


class DataLoader:

    def __init__(self, local_data_dir):

        self.local_data_dir = local_data_dir
        DataLoader._ensure_local_dir(self.local_data_dir)

    @staticmethod
    def _ensure_local_dir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            pass

    def load_data(self, data_path, header=None):

        if data_path.startswith("http"):

            downloaded_file = self._download_and_cache_data(data_url=data_path)
            local_file_suffix = downloaded_file.split(".")[-1]

            if local_file_suffix.startswith("data"):

                if header is not None:
                    return pd.read_csv(downloaded_file,
                                       header=0,
                                       sep=",")
                else:
                    return pd.read_csv(downloaded_file,
                                       header=0,
                                       sep=",")

            elif local_file_suffix.startswith("csv"):
                return pd.read_csv(downloaded_file,
                                   header=0,
                                   sep=";")
            else:
                raise DatasetFormatNotSupported
        else:
            raise MalformedDataUrl("%s is not a valid URL!" % (data_path, ))

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
                raise DataNotFoundRemotly("File %s not found at %s" %
                                          (save_file_name, data_url))
        return save_file_name


if __name__ == "__main__":

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

    data_loader = DataLoader(local_data_dir="./data")
    data = data_loader.load_data(data_path=URL)

    print(data.head())
