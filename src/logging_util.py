import sys
import logging
import argparse
from collections.abc import Iterable

import numpy as np
from tabulate import tabulate

try:
    import coloredlogs
except ImportError:
    print("for colored logs, run `pip install coloredlogs`")
    pass


class ColoredLog(object):

    """Docstring for ColoredLog. """

    def __init__(self, logger_name, verbose=3, file_handler=None):
        self._verbose = verbose
        self._level = self.log_level()
        self.file_handler = file_handler
        self.logger = self.construct_logger(logger_name)

    def log_level(self):
        if self._verbose == 0:
            self.level = 50
        elif self._verbose == 1:
            self.level = 40
        elif self._verbose == 2:
            self.level = 30
        elif self._verbose == 3:
            self.level = 20
        else:
            self.level = 10

    def construct_logger(self, logger_name):
        logger = logging.getLogger(logger_name)
        if "coloredlogs" in sys.modules:
            coloredlogs.DEFAULT_FIELD_STYLES["asctime"] = {"faint": True}
            coloredlogs.DEFAULT_FIELD_STYLES["name"] = {"faint": True}
            coloredlogs.DEFAULT_FIELD_STYLES["levelname"] = {"faint": True}
            coloredlogs.DEFAULT_LEVEL_STYLES["info"] = {"color": "blue"}
            coloredlogs.DEFAULT_LEVEL_STYLES["error"] = {"color": "magenta"}
            coloredlogs.install(
                level=self.level,
                logger=logger,
                fmt="%(asctime)s - %(name)s - %(levelname)s\n%(message)s\n",
                datefmt="%H:%M:%S",
            )
        else:
            logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(self.level)
            ch.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s\n%(message)s\n",
                    datafmt="%H:%M:%S",
                )
            )
            logger.addHandler(ch)
        return logger

    def table_message(
        self, data, caption="", transpose=False, header="firstrow", index=False
    ):
        if transpose:
            data = np.transpose(data)

        table_str = tabulate(
            data, headers=header, showindex=index, tablefmt="github", floatfmt=".4f"
        )

        printable = "\n".join([caption, table_str])
        return printable

    def debug(self, data, **kwargs):
        if isinstance(data, str):
            self.logger.debug(data)
        elif isinstance(data, Iterable):
            self.logger.debug(self.table_message(data, **kwargs))

    def info(self, data, **kwargs):
        if isinstance(data, str):
            self.logger.info(data)
        elif isinstance(data, Iterable):
            self.logger.info(self.table_message(data, **kwargs))

    def warning(self, data, **kwargs):
        if isinstance(data, str):
            self.logger.warning(data)
        elif isinstance(data, Iterable):
            self.logger.warning(self.table_message(data, **kwargs))

    def error(self, data, **kwargs):
        if isinstance(data, str):
            self.logger.error(data)
        elif isinstance(data, Iterable):
            self.logger.error(self.table_message(data, **kwargs))

    def critical(self, data, **kwargs):
        if isinstance(data, str):
            self.logger.critical(data)
        elif isinstance(data, Iterable):
            self.logger.critical(self.table_message(data, **kwargs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()
    logger = ColoredLog(__name__, args.verbose)

    logger.debug("DEBUG")
    logger.info("INFO")
    logger.warning("WARNING")
    logger.error("ERROR")
    logger.critical("CRITICAL")
    dat = [["1", "2", "3"], [4, 5, 6], [7, 8]]
    logger.info(dat, header=["a", "b", "c"], index=[1, 2, 3], caption="Test")
    #  printt(dat, caption="a table", header=["a", "b", "c"], index=[9, 8, 7])
