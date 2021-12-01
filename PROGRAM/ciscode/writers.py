import numpy as np
from pathlib import Path
import logging
log = logging.getLogger(__name__)


class Writer:
    """Abstract output formatter class."""

    def __init__(self, fname: str):
        self.fname = fname

    def save(self, output_dir: str = "."):
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        with open(output_dir / self.fname, "w") as file:
            file.write(str(self))
        log.info(f"Saved output to {output_dir / self.fname}")


class PA3(Writer):
    """Output formatter class for programming assignment 1."""

    def __init__(self, name: str, arr):
        super().__init__(f"{name}-Output.txt")
        self.name = name
        self.arr = arr

    def __str__(self):
        outputs = []
        name = self.name
        header = name + "-Output.txt"
        outputs.append(f"15,{header},0")
        arr = self.arr
        # logging.info("arr[0]")
        # logging.info(arr[0])

        for i in range(15):
            outputs.append(" ".join(map(lambda x: f"  {x:.03f}", arr[i])))

        # outputs.append(" ".join(map(lambda x: f"  {x:.02f}", self.d)))

        # outputs.append(
        #     " ".join(map(lambda x: f"  {x:.02f}", self.arr)))
        # for j in range(15):
        #     " ".join(map(lambda x: f"  {x:.02f}", self.arr[j]))
        # outputs += [
        #     # ", ".join(map(lambda x: f"  {x:.02f}", self.arr[k, i]))
        #     ", ".join(map(lambda x: f"  {x:.02f}", self.arr[k]))
        #     for k in range(15)
        #     # for i in range(7)
        # ]
        return "\n".join(outputs)
        # return (outputs)
