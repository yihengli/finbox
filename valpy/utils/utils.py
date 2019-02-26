from tqdm import tqdm, tqdm_notebook
import logging
import colorlog
import colorama
import os


def in_ipynb():
    """
    Check if the current environment is IPython Notebook

    Note, Spyder terminal is also using ZMQShell but cannot render Widget.

    Returns
    -------
    bool
        True if the current env is Jupyter notebook
    """
    try:
        zmq_status = str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>" # noqa E501
        spyder_status = any('SPYDER' in name for name in os.environ)
        return zmq_status and not spyder_status

    except NameError:
        return False


def get_tqdm():
    """
    Get proper tqdm progress bar instance based on if the current env is
    Jupyter notebook

    Note, Windows system doesn't supprot default progress bar effects

    Returns
    -------
    type, bool
        either tqdm or tqdm_notebook, the second arg will be ascii option
    """
    ascii = True if os.name == 'nt' else False

    if in_ipynb():
        # Jupyter notebook can always handle ascii correctly
        return tqdm_notebook, False
    else:
        return tqdm, ascii


class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        colorlog.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def get_logger(name='valpy'):
    logger = colorlog.getLogger(name)

    # Disable colorama if in jupyter notebook env
    if in_ipynb():
        colorama.deinit()

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = TqdmHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(log_color)s %(levelname)-6s%(reset)s %(white)s%(message)s')) # noqa E501
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
