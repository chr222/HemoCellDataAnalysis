from dataclasses import dataclass
from typing import get_type_hints, Dict


COLUMN_TYPE: Dict[type, str] = {
    int: 'integer',
    float: 'real'
}

RELATION_TYPE: Dict[type, str] = {
    Dict: 'dict'
}


def to_column(key: str, value_type: type):
    try:
        return f"{key} {COLUMN_TYPE[value_type]}"
    except KeyError:
        pass

    try:
        return f"{key} {RELATION_TYPE[value_type]}"
    except KeyError:
        return f"{key} UNKNOWN"

class EntityInterface:
    def __init__(self):
        self.id = None

    # def insert(self, connection, simulation_id: int):
    #     variables = vars(self)
    #     keys = list(filter(lambda k: k not in ['id', 'blocks'], variables.keys()))
    #
    #     variables_ = get_type_hints(self)
    #
    #     self.id = connection.insert(f"""INSERT INTO config (simulation_id, nx, ny, nz, warmup, stepMaterialEvery, stepParticleEvery, fluidEnvelope, rhoP, nuP, dx, dt, refDir, refDirN, blockSize, kBT, Re, particleEnvelope, kRep, RepCutoff, tmax, tmeas, tcsv, tcheckpoint, freq, coneAngle) VALUES (
    #             '{simulation_id}',
    #             '{self.nx}',
    #             '{self.ny}',
    #             '{self.nz}',
    #             '{self.warmup}',
    #             '{self.stepMaterialEvery}',
    #             '{self.stepParticleEvery}',
    #             '{self.fluidEnvelope}',
    #             '{self.rhoP}',
    #             '{self.nuP}',
    #             '{self.dx}',
    #             '{self.dt}',
    #             '{self.refDir}',
    #             '{self.refDirN}',
    #             '{self.blockSize}',
    #             '{self.kBT}',
    #             '{self.Re}',
    #             '{self.particleEnvelope}',
    #             '{self.kRep}',
    #             '{self.RepCutoff}',
    #             '{self.tmax}',
    #             '{self.tmeas}',
    #             '{self.tcsv}',
    #             '{self.tcheckpoint}',
    #             '{self.freq}',
    #             '{self.coneAngle}'
    #         );""")


    @classmethod
    def get_schema(cls) -> str:
        variables = get_type_hints(cls, include_extras=True)

        table = cls.__name__.lower()
        columns = [to_column(*item) for item in variables.items()]

        # query_new = f"CREATE TABLE {table} (id integer PRIMARY KEY, {});"

        query = """CREATE TABLE config (
                id integer PRIMARY KEY,
                simulation_id integer,
                nx integer,
                ny integer,
                nz integer,
                warmup integer,
                stepMaterialEvery integer,
                stepParticleEvery integer,
                fluidEnvelope integer,
                rhoP integer,
                nuP real,
                dx real,
                dt real,
                refDir integer,
                refDirN integer,
                blockSize integer,
                kBT real,
                Re real,
                particleEnvelope integer,
                kRep real,
                RepCutoff real,
                tmax integer,
                tmeas integer,
                tcsv integer,
                tcheckpoint integer,
                freq real,
                coneAngle real,
                FOREIGN KEY (simulation_id) REFERENCES simulation (id) ON DELETE CASCADE
            );"""

        return query