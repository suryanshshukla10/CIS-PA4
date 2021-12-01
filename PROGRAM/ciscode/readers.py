import numpy as np
from pathlib import Path
import logging


class Vertices:
    """Read vertices
    """

    def __init__(self, path: str):
        self.path = path
        with open(path, "r") as f:
            line = next(f)
            toks = line.replace(" ", "").split(",")

            self.N_vertices = int(toks[0])

        arrVer = np.loadtxt(path, delimiter=" ", skiprows=1,
                            max_rows=self.N_vertices)
        self.arrVer = arrVer


class Indices:
    """Read Indices
    """

    def __init__(self, path: str):
        self.path = path
        with open(path, "r") as f:
            line = next(f)
            toks = line.replace(" ", "").split(",")

            self.N_vertices = int(toks[0])

        arrInd = np.loadtxt(path, delimiter=" ", skiprows=self.N_vertices+2, usecols=(0, 1, 2)
                            )
        self.arrInd = arrInd


class RigidBody:
    """Parses the rigid body A and body B LED and tip coordinate in body coordinate: data."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "r") as f:
            line = next(f)
            toks = line.replace(" ", "").split(",")

        arr = np.loadtxt(path, skiprows=1)
        Y = arr[:6]
        self.Y = Y
        tip = arr[-1]
        self.tip = tip


class sampleReading:
    """Parses the sample reading data."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "r") as f:
            line = next(f)
            toks = line.replace(" ", "").split(",")
        arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        # NA1 = arr[:6]
        # self.NA1 = NA1
        # NB1 = arr[6:12]
        # self.NB1 = NB1
        # NA1 = []
        # NA1 = arr[:6]
        # NA1.append(12)
        # self.NA1 = NA1
        NA_dict = {}
        NB_dict = {}
        NA_dict[0] = arr[0:6]
        NB_dict[0] = arr[6:12]
        keys = range(14)
        for i in keys:
            k = i + 1
            x_a = k * 10 + k*6
            y_a = k*10 + (k+1)*6
            x_b = y_a
            y_b = x_b+6
            NA_dict[i+1] = arr[x_a:y_a]
            NB_dict[i+1] = arr[x_b:y_b]

        # logging.info(NA_dict[0])
        self.NA_dict = NA_dict
        self.NB_dict = NB_dict
