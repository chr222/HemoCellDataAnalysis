from src.linalg import Vector3Int

from dataclasses import dataclass
from typing import Dict


@dataclass
class Block:
    atomic_block: int
    size: Vector3Int
    offset: Vector3Int
    id: int = None

    def insert(self, connection, config_id: int):
        self.id = connection.insert(f"""INSERT INTO block (config_id, atomic_block, size_x, size_y, size_z, offset_x, offset_y, offset_z) VALUES (
            '{config_id}',
            '{self.atomic_block}',
            '{self.size.x}',
            '{self.size.y}',
            '{self.size.z}',
            '{self.offset.x}',
            '{self.offset.y}',
            '{self.offset.z}'
        );""")

    @staticmethod
    def get_schema() -> str:
        return """CREATE TABLE block (
            id integer PRIMARY KEY,
            config_id integer,
            atomic_block integer,
            size_x integer NOT NULL,
            size_y integer NOT NULL,
            size_z integer NOT NULL,
            offset_x integer NOT NULL,
            offset_y integer NOT NULL,
            offset_z integer NOT NULL,
            FOREIGN KEY (config_id) REFERENCES config (id) ON DELETE CASCADE 
        );"""


def load_blocks(connection, config_id: int) -> Dict[str, Block]:
    results = connection.select_all(f"""
        SELECT atomic_block, size_x, size_y, size_z, offset_x, offset_y, offset_z, id
        FROM block 
        WHERE config_id='{config_id}'
    """)

    blocks = {}
    for row in results:
        blocks[row[0]] = Block(row[0], Vector3Int(*row[1:4]), Vector3Int(*row[4:7]), row[7])

    return blocks
