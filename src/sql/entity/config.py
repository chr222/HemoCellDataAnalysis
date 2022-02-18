from bs4 import BeautifulSoup
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Dict, TYPE_CHECKING

import params
from src.sql.entity.block import Block, get_domain_info
from src.sql.entity import Entity, exclude, parent

if TYPE_CHECKING:
    from src.sql.connection import Connection


@dataclass
class Config(Entity):
    simulation_id: Annotated[int, parent('simulation', 'id')]

    # Required
    dx: float
    dt: float

    # Optional
    warmup: int
    stepMaterialEvery: int
    stepParticleEvery: int
    Re: float
    kRep: float
    RepCutoff: float
    tmax: int

    # Required (retrieved by get_domain_info)
    nx: int = None
    ny: int = None
    nz: int = None
    blocks: Annotated[Dict[str, Block], exclude] = None


def create_config(connection: "Connection", simulation_id: int, data_directory: Path, config_path: Path = None) -> Config:
    if config_path is None:
        config_path = data_directory / "config.xml"

    with open(config_path, 'r') as f:
        data = BeautifulSoup(f.read(), 'xml')

    config_params = {}
    for config_field, entity_field in [('dx', 'dx'), ('dt', 'dt')] + params.CONFIG_FIELDS:
        tag = data.find(config_field)

        if tag is None:
            continue

        value_type = Config.get_property_type(entity_field)
        config_params[entity_field] = value_type(tag.text.strip())

    config = Config(simulation_id=simulation_id, **config_params)

    blocks, domain = get_domain_info(data_directory)
    config.blocks = blocks
    config.nx = int(domain[0])
    config.ny = int(domain[1])
    config.nz = int(domain[2])

    config.insert(connection)

    return config


def load_config(connection: "Connection", simulation_id: int) -> Config:
    return Config.load(connection, simulation_id)
