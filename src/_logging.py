"""
Configures logging via Loguru: https://loguru.readthedocs.io/en/stable/
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


LOG_DIR = Path(__file__).parent.parent / "logs"


def configure_logging(name: Optional[str] = None, console: bool = True) -> None:
    """
    Configures logging to file, as well as optionally to the console.

    Parameters
    ----------
    name : Optional[str], default: None
        The module name. When provided, will save logs to file
        `"<root>/logs/{time}_{name}.log"`.
    console: bool, default: True
        Whether to log to the console.
    """
    logger.remove()

    if name:
        filename = "{time}_" + name + ".log"
        logger.add(LOG_DIR / filename, level="DEBUG")
    if console:
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green>  <level>{message}</level>",
        )
    return
