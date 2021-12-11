from math import dist
import click
import logging
from rich.logging import RichHandler
from rich.progress import track
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

from ciscode import readers, pointTriangleDistance2, triangleMesh, pa1Functions, writers, icp, closestPoint

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("ciscode")


@click.command()
@click.option("-d", "--data-dir", default="data", help="Where the data is.")
@click.option("-o", "--output_dir", default="outputs", help="Where to store outputs.")
# @click.option("-n", "--name", default="pa3-debug-a", help="Which experiment to run.")
def main(
    data_dir: str = "data", output_dir: str = "outputs", name: str = "-Debug-SampleReadingsTest.txt"
):
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.exists():
        output_dir.mkdir()
      ############Input data files############

    ##Surface Mesh Data structure##
    Vertices = readers.Vertices(data_dir / f"Problem4MeshFile.sur")
    logging.info(
        "loading triangle mesh  vertices data complete............................. ")
    Indices = readers.Indices(data_dir / f"Problem4MeshFile.sur")
    logging.info(
        "loading triangle mesh indices data complete.............................")
  ##Body Defination Files##

    # reading Rigid bodyA and B
    RigidBodyA = readers.RigidBody(data_dir / f"Problem4-BodyA.txt")
    logging.info(
        "loading Rigid Body A data complete.............................")
    RigidBodyB = readers.RigidBody(data_dir / f"Problem4-BodyB.txt")
    logging.info(
        "loading Rigid Body B data complete.........................")
    PA1 = pa1Functions.PA1
######Define the file name here ############################################
    # temp3 = ['A', 'B', 'C', 'D', 'E', 'F']
    # for q in track(temp3):

    # Unknown File 1
    input_file = "PA4-G-Unknown"
    complete_file_name = "PA4-G-Unknown-SampleReadingsTest.txt"

    # Unknown File 2
    # input_file = "PA4-H-Unknown"
    # complete_file_name = "PA4-H-Unknown-SampleReadingsTest.txt"

    # Unknown File 3
    # input_file = "PA4-J-Unknown-"
    # complete_file_name = "PA4-J-Unknown-SampleReadingsTest.txt"

    # Unknown File 4
    # input_file = "PA4-K-Unknown-"
    # complete_file_name = "PA4-K-Unknown-SampleReadingsTest.txt"
#####################################
    # Reading sampleReading.txt
    sampleReading = readers.sampleReading(data_dir / f"{complete_file_name}")

    # logging.info("NA[0] frame 0")
    # logging.info(sampleReadingA.NB_dict[0])
    logging.info("Debug-SampleReading reading complete...............")

    # Registration F_ak and F_bk
    logging.info("........Registration....F_a,k...")
    point_cloud_sk = []
    point_cloud_ck = []
    for k in range(75):
        print("\t \t ------------------------------Frame",
              k+1, "------------------------------")
        a = RigidBodyA.Y
        t_a = sampleReading.NA_dict[k]
        F_ak = PA1.points_registration(t_a, a)  # degine F_ak
        # logging.info(t_a.shape)
    # calculating F_bk
        b = RigidBodyB.Y
        t_b = sampleReading.NB_dict[k]
        # logging.info("t_b")
        # logging.info(t_b)
        F_bk = PA1.points_registration(t_b, b)
        # logging.info("..F_bk...matrix....")
        # logging.info(F_bk)

    # Pointer Tip with respect to rigid body B
        # logging.info("........d_k............")
        R_ak = F_ak[0:3, 0:3]
        P_ak = F_ak[0:3, 3]
        tip = RigidBodyB.tip
        # logging.info("loading tip")
        # logging.info(tip)

        # logging.info("loading R_ak")
        # logging.info(R_ak)

        # logging.info("loading P_ak")
        # logging.info(P_ak)
        temp1 = np.matmul(R_ak, tip) + P_ak

        # logging.info("RA+P")
        # logging.info(temp1)

        # inverse of F_bk
        R_bk = F_bk[0:3, 0:3]
        P_bk = F_bk[0:3, 3]
        R_bk_trans = np.transpose(R_bk)

        d_k = np.matmul(R_bk_trans, temp1) - np.matmul(R_bk_trans, P_bk)
        # d_k = d_k.T
        # logging.info(d_k.shape)
        # logging.info("d_k = F_bk_inv * F_ak * A_tip")
        # logging.info(d_k)

        #########triangle mesh calculation#####
        # triangle from the mesh
        # Get triangle vertices from the mesh data
        findTriangleVertices = triangleMesh.findTriangleVertices
        V = findTriangleVertices.getTriVertice(
            Vertices.arrVer, Indices.arrInd, 0)
        # logging.info("Triangle vertice of 100th tri")
        # logging.info(V)

        v1 = V[0]
        v2 = V[1]
        v3 = V[2]
        TRI = np.array([v1, v2, v3])
        # logging.info(TRI)
        # TRI = np.array()

        # To determine the closest point on the triangle
        # it always returns closest point on the triangle
        closestPointTriangle = pointTriangleDistance2.closestPoint

        def FindClosestPointMesh(sk):
            """[It calculates the closest point in triangle mesh]

            Args:
                sk ([3x1 vector]): [description]

            Returns:
                [3x1]: [closest point]
            """
            distance = {}
            pp0 = {}
            # for i in track(range(3135)):
            for i in range(3135):
                V = findTriangleVertices.getTriVertice(
                    Vertices.arrVer, Indices.arrInd, i)
                v1 = V[0]
                v2 = V[1]
                v3 = V[2]
                TRI = np.array([v1, v2, v3])
                # logging.info(TRI)
                new_dist, new_pp0 = closestPointTriangle.pointTriangleDistance(
                    TRI, sk)
                distance[i] = new_dist
                pp0[i] = new_pp0
                # finding minimum distance
            minimum = min(distance.items(), key=lambda x: x[1])
            minimum_key = minimum[0]
            closest_point = pp0[minimum_key]
            return closest_point

        # Freg = np.identity(3)
        # s_k = np.matmul(Freg, d_k)
        s_k = d_k
        c_k = FindClosestPointMesh(s_k)

        ###Point cloud###

        point_cloud_sk.append(d_k)
        point_cloud_ck.append(c_k)
    pt1 = np.array(point_cloud_sk)  # pt1 is point cloud
    pt2 = np.array(point_cloud_ck)  # pt2 is point cloud
    # logging.info(pt2.shape)
    # logging.info(pt2.shape)
    ICP = icp
    T, distances = ICP.icp(pt1, pt2, init_pose=None,
                           max_iterations=20, tolerance=0.001)
    # logging.info(T)
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    # logging.info(R)
    # logging.info(p)
    # logging.info(pt1[1])
    sk_new = []
    ck_new = []
    dist_new = []
    for i in range(75):
        ski = np.matmul(R, pt1[i]) + p
        cki = np.matmul(R, pt1[i]) + p
        dist_i = np.linalg.norm(ski - cki)

        sk_new.append(ski)
        ck_new.append(cki)
        dist_new.append(dist_i)

    sk_new = np.array(sk_new)
    ck_new = np.array(ck_new)
    dist_new = np.array(dist_new)

    out_list = []
    l1 = []
    for i in range(75):
        temp = [sk_new[i, 0], sk_new[i, 1], sk_new[i, 2],
                ck_new[i, 0], ck_new[i, 1], ck_new[i, 2], dist_new[i]]
        l1.append(temp)
    out_list = np.array(l1)

    output = writers.PA4(input_file, out_list)
    output.save(output_dir)


if __name__ == "__main__":
    main()
