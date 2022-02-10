from dataclasses import dataclass
import sys

from src.sql.entity.entity_interface import EntityInterface
from src.linalg import Vector3Int
from src.sql.entity.block import Block, load_blocks

import json
from typing import Annotated, Dict


@dataclass
class Config(EntityInterface):
    blocks: Annotated[Dict[str, Block], 'exclude']
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

    # Optionally add your custom parameters. The name should be the same as the one in the ConfigParams output
    freq: float
    coneAngle: float

    def insert(self, connection, simulation_id: int):
        variables = vars(self)
        keys = list(filter(lambda k: k not in ['id', 'blocks'], variables.keys()))

        self.id = connection.insert(f"""INSERT INTO config (simulation_id, nx, ny, nz, warmup, stepMaterialEvery, stepParticleEvery, fluidEnvelope, rhoP, nuP, dx, dt, refDir, refDirN, blockSize, kBT, Re, particleEnvelope, kRep, RepCutoff, tmax, tmeas, tcsv, tcheckpoint, freq, coneAngle) VALUES (
            '{simulation_id}',
            '{self.nx}',
            '{self.ny}',
            '{self.nz}',
            '{self.warmup}',
            '{self.stepMaterialEvery}',
            '{self.stepParticleEvery}',
            '{self.fluidEnvelope}',
            '{self.rhoP}',
            '{self.nuP}',
            '{self.dx}',
            '{self.dt}',
            '{self.refDir}',
            '{self.refDirN}',
            '{self.blockSize}',
            '{self.kBT}',
            '{self.Re}',
            '{self.particleEnvelope}',
            '{self.kRep}',
            '{self.RepCutoff}',
            '{self.tmax}',
            '{self.tmeas}',
            '{self.tcsv}',
            '{self.tcheckpoint}',
            '{self.freq}',
            '{self.coneAngle}'
        );""")

    # @staticmethod
    # def get_schema() -> str:
    #     variables_ = get_type_hints()
    #
    #     return """CREATE TABLE config (
    #         id integer PRIMARY KEY,
    #         simulation_id integer,
    #         nx integer,
    #         ny integer,
    #         nz integer,
    #         warmup integer,
    #         stepMaterialEvery integer,
    #         stepParticleEvery integer,
    #         fluidEnvelope integer,
    #         rhoP integer,
    #         nuP real,
    #         dx real,
    #         dt real,
    #         refDir integer,
    #         refDirN integer,
    #         blockSize integer,
    #         kBT real,
    #         Re real,
    #         particleEnvelope integer,
    #         kRep real,
    #         RepCutoff real,
    #         tmax integer,
    #         tmeas integer,
    #         tcsv integer,
    #         tcheckpoint integer,
    #         freq real,
    #         coneAngle real,
    #         FOREIGN KEY (simulation_id) REFERENCES simulation (id) ON DELETE CASCADE
    #     );"""


def create_config(connection, simulation_id: int, output_directory: str) -> Config:
    with open(f"{output_directory}/log/logfile", "r") as f:
        data = "".join([line for line in f.readlines()])

        try:
            result = connection.find_json_string("ConfigParams", data)
        except Exception:
            print(f"Missing the ConfigParams JSON object in {output_directory}/log/logfile", file=sys.stderr)
            exit(1)

        params = json.loads(result)
        config = Config(**params)
        config.insert(connection, simulation_id)

        config.blocks = {}
        for atomic_block, block in list(params.values())[0].items():
            config.blocks[atomic_block] = Block(atomic_block, Vector3Int(*block['size']), Vector3Int(*block['offset']))
            config.blocks[atomic_block].insert(connection, config.id)

        return config


def load_config(connection, simulation_id: int) -> Config:
    params = connection.select_one(f"""
        SELECT
            nx,
            ny,
            nz,
            warmup,
            stepMaterialEvery,
            stepParticleEvery,
            fluidEnvelope,
            rhoP,
            nuP,
            dx,
            dt,
            refDir,
            refDirN,
            blockSize,
            kBT,
            Re,
            particleEnvelope,
            kRep,
            RepCutoff,
            tmax,
            tmeas,
            tcsv,
            tcheckpoint,
            freq,
            coneAngle,
            id
        FROM config
        WHERE simulation_id='{simulation_id}';""")

    return Config(
        load_blocks(connection, params[-1]),
        *params
    )
