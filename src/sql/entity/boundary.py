from dataclasses import dataclass
import numpy as np
from scipy.signal import fftconvolve


@dataclass
class Boundary:
    boundary_map: np.ndarray
    id: int = None

    def insert(self, connection, simulation_id: int):
        self.id = connection.insert(
            f"INSERT INTO boundary (simulation_id, boundary_map) VALUES (?, ?);",
            (simulation_id, self.boundary_map)
        )

    @staticmethod
    def get_schema() -> str:
        return """CREATE TABLE boundary (
                    id integer PRIMARY KEY,
                    simulation_id integer,
                    boundary_map array,
                    FOREIGN KEY (simulation_id) REFERENCES simulation (id) ON DELETE CASCADE
                  );"""

    @staticmethod
    def is_already_created(connection, simulation_id: int) -> bool:
        return connection.exists(f"SELECT id FROM boundary WHERE simulation_id='{simulation_id}';")[0]

    @property
    def boundaries(self) -> np.ndarray:
        kernel = np.ones((3, 3, 3))
        boundaries = np.minimum(fftconvolve(self.boundary_map, kernel, mode='same'), 1.0)

        return np.where(boundaries == 1)


def load_boundary(connection, simulation_id: int) -> Boundary:
    return Boundary(*connection.select_one(f"SELECT boundary_map, id FROM boundary WHERE simulation_id='{simulation_id}';"))