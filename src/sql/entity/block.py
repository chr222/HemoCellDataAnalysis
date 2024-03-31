import sys
from dataclasses import dataclass
import h5py
import numpy as np
from pathlib import Path
from typing import Dict

from src.linalg import Vector3Int


@dataclass
class Block:
    atomic_block: int
    size: Vector3Int
    offset: Vector3Int


def get_domain_info(data_directory: Path) -> (Dict[str, Block], np.array):
    try:
        first_directory = sorted(list(data_directory.glob("hdf5/*")))[0]
    except IndexError:
        print("No HDF5 data directory found, so no Block data could be created", file=sys.stderr)
        return None

    file_paths = sorted(list(first_directory.glob("Fluid.*.p.*.h5")))

    if len(file_paths) == 0:
        print("No HDF5 Fluid files found, so no Block data could be created", file=sys.stderr)
        return None

    blocks: Dict[str, Block] = {}
    domain = np.zeros(3, dtype=int)
    for path in file_paths:
        # Get atomic block from file path
        atomic_block = path.name.split(".")[-2]

        with h5py.File(path, 'r') as f:
            dx = f.attrs['dx'][0]

            # Reversed since HemoCell outputs zyx order
            size = np.array(f.attrs['subdomainSize'] - 2, dtype=int)[::-1]

            # +1.5 since HemoCell does -1.5 for some reason
            offset = np.array(f.attrs['relativePosition'] / dx + 1.5, dtype=int)[::-1]

            # Calculate domain size
            domain = np.maximum(domain, size + offset)

            blocks[atomic_block] = Block(
                atomic_block=int(atomic_block),
                size=Vector3Int(*map(int, size)),
                offset=Vector3Int(*map(int, offset))
            )

    return blocks, domain
