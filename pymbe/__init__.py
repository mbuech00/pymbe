__all__ = ["MBE"]

from pymbe.pymbe import MBE

import sys
from mpi4py import MPI

from pymbe.logger import logger


# change exception handling if parallel calculation
if MPI.COMM_WORLD.Get_size() > 1:
    # get functions for handling exceptions
    sys_excepthook = sys.excepthook

    # define function that ensures exceptions are handled without deadlock
    def global_except_hook(exctype, value, traceback):
        DIVIDER = "*" * 93

        try:
            logger.error("\n" + DIVIDER + "\n")
            logger.error(
                "Uncaught exception was detected on rank {}.\n".format(
                    MPI.COMM_WORLD.Get_rank()
                )
            )
            sys_excepthook(exctype, value, traceback)
            logger.error("\nShutting down MPI processes...\n")
            logger.error(DIVIDER + "\n")
        finally:
            try:
                MPI.COMM_WORLD.Abort(1)
            except Exception as exception:
                logger.error("MPI failed to stop, this process will hang.\n")
                logger.error(DIVIDER + "\n")
                raise exception

    # set function for handling exceptions
    sys.excepthook = global_except_hook
