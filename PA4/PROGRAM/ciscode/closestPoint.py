from typing import Tuple
import numpy as np
from numpy import dot
from math import sqrt
import logging


class CP:
    def closestPoint_fun(V):
        v1 = V[0]
        v2 = V[1]
        v3 = V[2]
        TRI = np.array([v1, v2, v3])
