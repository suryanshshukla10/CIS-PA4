from typing import Tuple
import numpy as np
from numpy import dot
from math import sqrt
import logging


class closestPoint:
    def centroid(V1, V2, V3):
        """[summary]

        Args:
            V1 ([x1,y1,z1]): [vertex 1 of triangle]
            V2 ([x2,y2,z2]): [vertex 2 of triangle]
            V3 ([x3,y3,z3]): [vertex 3 of triangle]

        Returns:
            [x0,y0,z0]: [coordinates of the centroid of triangle]
        """

        #        V3
        #        |\
        #        | \
        #        |  \
        #        |   \
        #        |    \
        #        |     \
        #     V1 |      \ V2
        #        *-------*

        V1 = np.array(V1)
        V2 = np.array(V2)
        V3 = np.array(V3)

        x0 = (V1[0] + V2[0] + V3[0]) / 3
        y0 = (V1[1] + V2[1] + V3[1]) / 3
        z0 = (V1[2] + V2[2] + V3[2]) / 3
        C0 = [x0, y0, z0]
        # logging.info("centroid")
        # logging.info(C0)
        return C0

    def pointTriangleDistance(TRI, P):
        """[Calculate the distance of a given point P from a triangle TRI.]

        Args:
            TRI ([[P1;P2;P3]]): [The triangle is a matrix
    #   formed by three rows of points TRI = [P1;P2;P3] each of size 1x3]
            P ([P0 = [0.5 -0.3 0.5]]): [Point P is a row vector of the form 1x3]
        """
        #
        #        ^t
        #  \     |
        #   \reg2|
        #    \   |
        #     \  |
        #      \ |
        #       \|
        #        *P2
        #        |\
        #        | \
        #  reg3  |  \ reg1
        #        |   \
        #        |reg0\
        #        |     \
        #        |      \ P1
        # -------*-------*------->s
        #        |P0      \
        #  reg4  | reg5    \ reg6
        B = TRI[0, :]
        E0 = TRI[1, :] - B
        # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
        E1 = TRI[2, :] - B
        # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
        D = B - P
        a = dot(E0, E0)
        b = dot(E0, E1)
        c = dot(E1, E1)
        d = dot(E0, D)
        e = dot(E1, D)
        f = dot(D, D)
        # print "{0} {1} {2} ".format(B,E1,E0)
        det = a * c - b * b
        s = b * e - c * d
        t = b * d - a * e

        if (s + t) <= det:
            if s < 0.0:
                if t < 0.0:
                    # region4
                    if d < 0:
                        # logging.info("region4")
                        t = 0.0
                        if -d >= a:
                            s = 1.0
                            sqrdistance = a + 2.0 * d + f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
                    else:
                        s = 0.0
                        if e >= 0.0:
                            t = 0.0
                            sqrdistance = f
                        else:
                            if -e >= c:
                                t = 1.0
                                sqrdistance = c + 2.0 * e + f
                            else:
                                t = -e / c
                                sqrdistance = e * t + f

                                # of region 4
                else:
                    # region 3
                    # logging.info("Point in region 3")
                    s = 0
                    if e >= 0:
                        t = 0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f
                            # of region 3
            else:
                if t < 0:
                    # region 5
                    # logging.info("Point in region 5")
                    t = 0
                    if d >= 0:
                        s = 0
                        sqrdistance = f
                    else:
                        if -d >= a:
                            s = 1
                            sqrdistance = a + 2.0 * d + f  # GF 20101013 fixed typo d*s ->2*d
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
                else:
                    # region 0
                    # logging.info("Point in region 0")
                    invDet = 1.0 / det
                    s = s * invDet
                    t = t * invDet
                    sqrdistance = s * (a * s + b * t + 2.0 * d) + \
                        t * (b * s + c * t + 2.0 * e) + f
        else:
            if s < 0.0:
                # region 2
                # logging.info("Point in region 2")
                tmp0 = b + d
                tmp1 = c + e
                if tmp1 > tmp0:  # minimum on edge s+t=1
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f  # GF 20101014 fixed typo 2*b -> 2*d
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2 * d) + \
                            t * (b * s + c * t + 2 * e) + f

                else:  # minimum on edge s=0
                    s = 0.0
                    if tmp1 <= 0.0:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        if e >= 0.0:
                            t = 0.0
                            sqrdistance = f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f
                            # of region 2
            else:
                if t < 0.0:
                    # region6
                    # logging.info("region 6")
                    tmp0 = b + e
                    tmp1 = a + d
                    if tmp1 > tmp0:
                        numer = tmp1 - tmp0
                        denom = a - 2.0 * b + c
                        if numer >= denom:
                            t = 1.0
                            s = 0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = numer / denom
                            s = 1 - t
                            sqrdistance = s * \
                                (a * s + b * t + 2.0 * d) + t * \
                                (b * s + c * t + 2.0 * e) + f

                    else:
                        t = 0.0
                        if tmp1 <= 0.0:
                            s = 1
                            sqrdistance = a + 2.0 * d + f
                        else:
                            if d >= 0.0:
                                s = 0.0
                                sqrdistance = f
                            else:
                                s = -d / a
                                sqrdistance = d * s + f
                else:
                    # region 1

                    numer = c + e - b - d
                    if numer <= 0:
                        s = 0.0
                        t = 1.0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        denom = a - 2.0 * b + c
                        if numer >= denom:
                            s = 1.0
                            t = 0.0
                            sqrdistance = a + 2.0 * d + f
                        else:
                            s = numer / denom
                            t = 1 - s
                            sqrdistance = s * \
                                (a * s + b * t + 2.0 * d) + t * \
                                (b * s + c * t + 2.0 * e) + f

        # account for numerical round-off error
        if sqrdistance < 0:
            sqrdistance = 0

        dist = sqrt(sqrdistance)

        PP0 = B + s * E0 + t * E1
        # logging.info("distance:")
        # logging.info(dist)
        # logging.info("point coordinates")
        # logging.info(PP0)

        return dist, PP0
