import click
import logging
from rich.logging import RichHandler
from rich.progress import track
from pathlib import Path
import numpy as np


from ciscode import readers, pointTriangleDistance2, triangleMesh, pa1Functions, writers

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

    # Read inputs
    Vertices = readers.Vertices(data_dir / f"Problem3Mesh.sur")
    logging.info(
        "loading triangle mesh  vertices data complete............................. ")
    Indices = readers.Indices(data_dir / f"Problem3Mesh.sur")
    logging.info(
        "loading triangle mesh indices data complete.............................")

    # logging.info("Vertices")
    # logging.info(Vertices.arrVer)
    # logging.info("Indices")
    # logging.info(Indices.arrInd)

    # reading Rigid bodyA and B
    RigidBodyA = readers.RigidBody(data_dir / f"Problem3-BodyA.txt")
    logging.info(
        "loading Rigid Body A data complete.............................")
    RigidBodyB = readers.RigidBody(data_dir / f"Problem3-BodyB.txt")
    logging.info(
        "loading Rigid Body B data complete.........................")
######Define the file name here ############################################
    temp3 = ['A', 'B', 'C', 'D', 'E', 'F']
    for q in track(temp3):
        input_file = "PA3-"+q
        print("\t \t ------------------------------Input File: PA3-",
              q, "-Debug-SampleReadingsTest------------------------------")

        # Reading sampleReading.txt
        sampleReading = readers.sampleReading(data_dir / f"{input_file}{name}")

        # logging.info("NA[0] frame 0")
        # logging.info(sampleReadingA.NB_dict[0])
        logging.info("Debug-SampleReading reading complete...............")

        # Tip and Y values of rigid body
        # logging.info(RigidBodyA.tip)
        # logging.info(RigidBodyA.Y)
        # logging.info(RigidBodyB.tip)
        # logging.info(RigidBodyB.Y)

        # Registration
        PA1 = pa1Functions.PA1

        output_pts_dict = {}

        logging.info("........Registration....F_a,k...")
        # Running code for 15 the frames
        for k in range(15):
            print("\t \t ------------------------------Frame",
                  k+1, "------------------------------")
            a = RigidBodyA.Y
            t_a = sampleReading.NA_dict[k]
            F_ak = PA1.points_registeration(t_a, a)
            # logging.info("..F_ak...matrix....")
            # logging.info(F_ak)

            # logging.info("........Registration....F_b,k...")
            b = RigidBodyB.Y
            t_b = sampleReading.NB_dict[k]
            # logging.info("t_b")
            # logging.info(t_b)
            F_bk = PA1.points_registeration(t_b, b)
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

            # logging.info("d_k = F_bk_inv * F_ak * A_tip")
            # logging.info(d_k)

            # defining Freg matrix to identity for only problem 3
            Freg = np.identity(3)
            # logging.info("Freg=")
            # logging.info(Freg)

            # computing sk
            sk = np.matmul(Freg, d_k)
            # logging.info("sk = Freg*dk")
            # logging.info(sk)

            # computing ck
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
            closestPoint = pointTriangleDistance2.closestPoint

            # print("\t \tFinding closest point on triangle for k=", k+1)

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
                new_dist, new_pp0 = closestPoint.pointTriangleDistance(TRI, sk)
                distance[i] = new_dist
                pp0[i] = new_pp0

            # finding minimum distance
            minimum = min(distance.items(), key=lambda x: x[1])
            minimum_key = minimum[0]
            closest_point = pp0[minimum_key]
            # logging.info(minimum[0])
            # logging.info(pp0[minimum_key])
            d = closest_point
            c = closest_point
            d_c = d - c
            d_c_norm = np.linalg.norm(d_c)

            l1 = np.concatenate((d, c))
            l2 = list(l1)
            l2.append(d_c_norm)
            # logging.info(d_c_norm)
            # logging.info("output")
            # logging.info(l2)
            output_pts_dict[k] = l2

        out_list = []
        for l in range(15):
            out_list.append(output_pts_dict[l])
            # logging.info(output_pts_dict[0])
        out_list = np.array(out_list)
        # logging.info(out_list)
        # output = writers.PA3(name, l2)
        # logging.info(l2)

        output = writers.PA3(input_file, out_list)
        output.save(output_dir)


if __name__ == "__main__":
    main()
