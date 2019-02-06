from tqdm import tqdm, tqdm_notebook
import logging
import colorlog


def in_ipynb():
    """
    Check if the current environment is IPython Notebook

    Returns
    -------
    bool
        True if the current env is Jupyter notebook
    """
    try:
        return str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>" # noqa E501
    except NameError:
        return False


def get_tqdm():
    """
    Get proper tqdm progress bar instance based on if the current env is
    Jupyter notebook

    Returns
    -------
    type
        either tqdm or tqdm_notebook
    """

    if in_ipynb():
        return tqdm_notebook
    else:
        return tqdm


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        colorlog.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def get_logger(name='valpy'):
    logger = colorlog.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = TqdmHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(log_color)s %(levelname)-6s%(reset)s %(white)s%(message)s')) # noqa E501
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
