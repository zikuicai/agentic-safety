import colorama
from colorama import Fore, Back, Style

import logging

colorama.init()


class BaseColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages based on their level"""

    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Back.WHITE
    }

    def format(self, record):
        # Add color to the log level name and message
        color = self.COLORS.get(record.levelno, Fore.WHITE)
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


def setup_logger(name='colored_logger', log_file='app.log', level=logging.DEBUG):
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    console_handler = logging.StreamHandler()
    # file_handler = logging.FileHandler(log_file)

    # Set levels
    console_handler.setLevel(level)
    # file_handler.setLevel(level)

    # Create formatters
    console_formatter = BaseColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Add formatters to handlers
    console_handler.setFormatter(console_formatter)
    # file_handler.setFormatter(file_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)

    return logger
