import logging
import numpy as np
import logging


class findTriangleVertices:
    """[This class will find the triangle vertices from give indices and vertex]
    """
    def getTriVertice(verts, tInd, i):
        """[function for finding the triangle vertices coordinate]

        Args:
            verts ([array]): [all the vertex array]
            tInd ([array]): [all the indices array]
            i ([integer]): [triangle which to find for indices ]

        Returns:
            [V1,V2,V3]: [each of V1,V2,V3 contains the xy,z coordinate of each of the  vertices of triangle]
        """

        triangle = tInd[i]

        V1 = verts[int(triangle[0])]
        V2 = verts[int(triangle[1])]
        V3 = verts[int(triangle[2])]

        return [V1, V2, V3]
