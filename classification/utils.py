import logging

_ch = logging.StreamHandler()
_ch.setLevel(logging.DEBUG)
_ch.setFormatter(logging.Formatter('(%(module)s) [%(levelname)s] - %(message)s'))


def get_logger(lv=logging.INFO):
    """Get the default logger.

    Args:
        lv:
            Logging level.

    Returns:
        Logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(lv)
    logger.addHandler(_ch)
    return logger


_logger = get_logger()
