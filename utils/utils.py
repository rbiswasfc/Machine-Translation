import logging
import string
import functools
import unicodedata
import traceback
import pandas as pd
from functools import wraps
from pathlib import Path


def setup_logger(logger):
    """
    configure logger for the project
    
    :param logger: a logger object
    :type logger: logging.Logger
    :return: logger with desired settings
    :rtype: logging.Logger
    """
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler("project_execution.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def try_except(logger):
    """
    A decorator for exception handing
    :param logger: logger to used inside the docorator
    :type logger: logging.Logger
    :return a docorator
    :rtype: function
    """

    def main_decorator(func):
        """
        Function containing decorator logic
        :param func: function for which exception handing will be performed
        :type func: function
        :return: wrapped function
        :rtype: function
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                value = func(*args, **kwargs)
                return value
            except Exception:
                # traceback.print_exc()
                logger.exception("Exception in running {}".format(func.__name__))
                return None

        return wrapper

    return main_decorator


def human_format(num, suffix="B"):
    """
    human readable format for file sizes
    :param num: file size in bytes
    :type num: int
    :param suffix: suffix to use for end result, defaults to 'B'
    :type suffix: str, optional
    :return: human readable file size
    :rtype: str
    """
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if num < 1024.0:
            return "{:.1f} {}{}".format(num, unit, suffix)
        num /= 1024.0
    return "{:.1f} {}{}".format(num, "Y", suffix)


def get_dataframe_memory_usage(df):
    """
    get in-memory size of a pandas dataframe
    :param df: input dataframe
    :type df: pd.DataFrame
    :return: human readable dataframe size
    :rtype: str
    """
    s = df.memory_usage(deep=True).sum()
    return human_format(s)


def check_create_dir(dir_path):
    """
    check if folder exists at a specific path, if not create the directory
    :param dir_path: path to the directory to be created
    :type dir_path: str
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)
