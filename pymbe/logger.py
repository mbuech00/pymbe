#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
logger module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import sys
import logging


# add custom logger class
class PyMBELogger(logging.Logger):
    def info2(self, msg: str, *args, **kwargs) -> None:
        if self.isEnabledFor(logging.INFO - 5):
            self._log(logging.INFO - 5, msg, args, **kwargs)


# get logger
logger = PyMBELogger("pymbe_logger")

# remove handlers from possible previous initialization
if logger.hasHandlers():
    logger.handlers.clear()

# add new handler to log to stdout
handler = logging.StreamHandler(sys.stdout)

# create new formatter
formatter = logging.Formatter("%(message)s")

# add formatter to handler
handler.setFormatter(formatter)

# add handler to logger if it does not already exist
logger.addHandler(handler)

# prevent logger from propagating handlers from parent loggers
logger.propagate = False
