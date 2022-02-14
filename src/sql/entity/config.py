from dataclasses import dataclass
import sys
import json
from typing import Annotated, Dict, TYPE_CHECKING, List

from src.linalg import Vector3Int
from src.sql.entity.block import Block
from src.sql.entity import Entity, exclude, parent

if TYPE_CHECKING:
    from src.sql.connection import Connection


@dataclass
class Config(Entity):
    simulation_id: Annotated[int, parent('simulation', 'id')]
    nx: int
    ny: int
    nz: int
    warmup: int
    stepMaterialEvery: int
    stepParticleEvery: int
    fluidEnvelope: int
    rhoP: int
    nuP: float
    dx: float
    dt: float
    refDir: int
    refDirN: int
    blockSize: int
    kBT: float
    Re: float
    particleEnvelope: int
    kRep: float
    RepCutoff: float
    tmax: int
    tmeas: int
    tcsv: int
    tcheckpoint: int

    # Excluded data
    blocks_data: Annotated[Dict[str, Block], exclude] = None


def create_config(connection: "Connection", simulation_id: int, data_directory: str) -> Config:
    with open(f"{data_directory}/log/logfile", "r") as f:
        data = "".join([line for line in f.readlines()])

        try:
            result = connection.find_json_string("ConfigParams", data)
        except Exception:
            print(f"Could not parse the ConfigParams JSON object in {data_directory}/log/logfile", file=sys.stderr)
            exit(1)

        params = json.loads(result)
        config = Config.from_dict(simulation_id=simulation_id, **params)
        config.insert(connection)

        if 'blocks' in params:
            config.blocks_data = {}
            for atomic_block, block in params['blocks'].items():
                config.blocks_data[atomic_block] = Block(
                    config_id=config.id,
                    atomic_block=atomic_block,
                    size=Vector3Int(*block['size']),
                    offset=Vector3Int(*block['offset'])
                )
                config.blocks_data[atomic_block].insert(connection)
        else:
            print(f"Missing blocks info in the ConfigParams JSON object in {data_directory}/log/logfile")

        return config


def load_config(connection: "Connection", simulation_id: int) -> Config:
    config: Config = Config.load(connection, simulation_id)

    blocks: List[Block] = Block.load_all(connection, config.id)

    if len(blocks):
        config.blocks_data = {block.atomic_block: block for block in blocks}
    else:
        print("This simulation is missing blocks data. As a result the HDF5 cannot be parsed.", file=sys.stderr)

    return config
