from src.sql.entity import parent, Entity
from src.linalg import Vector3Int

from dataclasses import dataclass
from typing import Annotated


@dataclass
class Block(Entity):
    config_id: Annotated[int, parent('config', 'id')]
    atomic_block: int
    size: Vector3Int
    offset: Vector3Int
