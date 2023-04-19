__all__ = ["MBE"]

from pymbe.pymbe import MBE

import sys
import logging

# get logger
logger = logging.getLogger("pymbe_logger")

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

# get functions for handling exceptions
sys_excepthook = sys.excepthook


# define function that ensures exceptions are handled without deadlock
def global_except_hook(exctype, value, traceback):
    import mpi4py.MPI

    DIVIDER = "*" * 93

    try:
        logger.error("\n" + DIVIDER + "\n")
        logger.error(
            "Uncaught exception was detected on rank {}.\n".format(
                mpi4py.MPI.COMM_WORLD.Get_rank()
            )
        )
        sys_excepthook(exctype, value, traceback)
        logger.error("\nShutting down MPI processes...\n")
        logger.error(DIVIDER + "\n")
    finally:
        try:
            mpi4py.MPI.COMM_WORLD.Abort(1)
        except Exception as exception:
            logger.error("MPI failed to stop, this process will hang.\n")
            logger.error(DIVIDER + "\n")
            raise exception


# set function for handling exceptions
sys.excepthook = global_except_hook
