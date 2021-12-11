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

    ############Input data files###########################
    ##Surface Mesh Data structure##
    Vertices = readers.Vertices(data_dir / f"Problem5MeshFile.sur")
    logging.info(".....Loading Triangle Mesh......... ")

    #########triangle vetices##############################
    vertices = Vertices.arrVer
    logging.info(vertices[1])
    # logging.info(type(vertices))


if __name__ == "__main__":
    main()
