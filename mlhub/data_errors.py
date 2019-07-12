class DataLoadingError(Exception):
    pass


class DataNotFoundRemotly(DataLoadingError):
    pass


class DatasetFormatNotSupported(DataLoadingError):
    pass


class MalformedDataUrl(DataLoadingError):
    pass
