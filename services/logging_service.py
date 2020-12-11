import logging
from logging.handlers import TimedRotatingFileHandler

FORMMATER = logging.Formatter("%(asctime)s:%(name)8s:%(levelname)s:%(message)s")


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(FORMMATER)
    stream_handler.setLevel(logging.DEBUG)
    time_rotating_hndlr = TimedRotatingFileHandler("logs/logs.log", 'D')
    time_rotating_hndlr.setFormatter(FORMMATER)
    time_rotating_hndlr.setLevel(logging.WARN)
    logger.addHandler(stream_handler)
    logger.addHandler(time_rotating_hndlr)
    return logger
