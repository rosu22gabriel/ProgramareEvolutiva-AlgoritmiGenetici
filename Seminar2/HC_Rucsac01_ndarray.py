import numpy as np
from typing import Tuple, List, Union

# Definirea tipurilor pentru claritate in ndarray
# ndarray poate contine intregi sau float-uri in functie de context
VectorInt = np.ndarray
VectorFloat = np.ndarray
MatrixInt = np.ndarray

def ok(x: VectorInt, c: VectorFloat, v: VectorFloat, maxim: float) -> Tuple[bool, float]:
    """
    Verifica daca solutia x este fezabila (nu depaseste capacitatea) si calculeaza fitness-ul.

    Args:
        x: Vector binar (0/1)
    """