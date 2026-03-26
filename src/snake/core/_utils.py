"""Utils module for internal use."""


class DuplicateFilter:
    """
    Filters away duplicate log messages.

    Modified version of: https://stackoverflow.com/a/31953563/965332.
    """

    def __init__(self, logger):  # noqa: ANN001
        self.msgs = set()
        self.logger = logger

    def filter(self, record) -> bool:  # noqa: ANN001
        msg = str(record.msg)
        is_duplicate = msg in self.msgs
        if not is_duplicate:
            self.msgs.add(msg)
        return not is_duplicate

    def __enter__(self):
        self.logger.addFilter(self)

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self.logger.removeFilter(self)
