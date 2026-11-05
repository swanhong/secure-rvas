import ctypes
import subprocess
from pathlib import Path

import numpy as np


class Davies:
    def __init__(self) -> None:
        library_directory = Path(
            subprocess.check_output(
                [
                    "Rscript",
                    "-e",
                    'cat(system.file("libs", package="SKAT", mustWork=TRUE))',
                ],
                text=True,
            ).strip()
        )
        library = sorted(library_directory.glob("SKAT.*"))[0]
        self.function = ctypes.CDLL(str(library)).qfc
        self.function.argtypes = [
            np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # lambda
            np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # delta
            np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"),  # degree
            ctypes.POINTER(ctypes.c_int),  # nlambda
            ctypes.POINTER(ctypes.c_double),  # sigma
            ctypes.POINTER(ctypes.c_double),  # q
            ctypes.POINTER(ctypes.c_int),  # lim
            ctypes.POINTER(ctypes.c_double),  # acc
            np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),  # array
            ctypes.POINTER(ctypes.c_int),  # ifault
            ctypes.POINTER(ctypes.c_double),  # p-value
        ]
        self.function.restype = None

    def p_value(
        self,
        statistic: float,
        eigenvalues: np.ndarray,
        modified_liu_p: float,
    ) -> tuple[float, int]:
        if eigenvalues.size == 1:
            return modified_liu_p, 1

        lambdas = np.ascontiguousarray(eigenvalues, dtype=np.float64)
        noncentral = np.zeros(lambdas.size, dtype=np.float64)
        degrees = np.ones(lambdas.size, dtype=np.int32)
        count = ctypes.c_int(lambdas.size)
        sigma = ctypes.c_double(0)
        q = ctypes.c_double(statistic)
        limit = ctypes.c_int(10000)
        accuracy = ctypes.c_double(1e-6)
        trace = np.zeros(7, dtype=np.float64)
        fault = ctypes.c_int(0)
        cumulative = ctypes.c_double(0)
        self.function(
            lambdas,
            noncentral,
            degrees,
            ctypes.byref(count),
            ctypes.byref(sigma),
            ctypes.byref(q),
            ctypes.byref(limit),
            ctypes.byref(accuracy),
            trace,
            ctypes.byref(fault),
            ctypes.byref(cumulative),
        )
        p_value = 1 - cumulative.value
        converged = int(fault.value == 0)
        if p_value > 1 or p_value <= 0:
            return modified_liu_p, 0
        return p_value, converged
