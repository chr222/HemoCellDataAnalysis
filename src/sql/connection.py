from src.sql.entity.boundary import Boundary
from src.sql.entity.config import Config
from src.sql.entity.csv_cell import CSVCell
from src.sql.entity.csv_iteration import CSVIteration
from src.sql.entity.hdf5_cell import Hdf5Cell
from src.sql.entity.hdf5_fluid import HDF5Fluid
from src.sql.entity.hdf5_iteration import Hdf5Iteration
from src.sql.entity.simulation import Simulation

from glob import glob
from math import ceil
import numpy as np
from os import mkdir
from os.path import exists, isfile, isdir
from scipy.io import savemat, loadmat
import sqlite3
import sys
from typing import Union
from uuid import uuid4


class Connection:
    connection: sqlite3.dbapi2

    def __init__(self, database_name: str, matrix_directory: str):
        # Setup handler for numpy arrays
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        sqlite3.register_converter("array", self.convert_array)

        db_exists = self.database_exists(database_name)

        self.connection = sqlite3.connect(database_name, timeout=3600, detect_types=sqlite3.PARSE_DECLTYPES)

        if not db_exists:
            print("\rDatabase does not exist yet. Creating tables...", end="")
            self.create_schema()
            print("\rSuccessfully created tables")

        # Matrices are saved in a separate directory
        self.matrix_directory = matrix_directory
        if not isdir(matrix_directory):
            print("\rMatrix directory does not exist yet", end="")
            mkdir(matrix_directory)
            print("\rCreated matrix directory")


    @staticmethod
    def database_exists(database_name: str) -> bool:
        return exists(database_name) and isfile(database_name)

    def create_schema(self) -> sqlite3.dbapi2:
        cursor = self.connection.cursor()

        for entity in [Simulation, Config, Hdf5Iteration, Hdf5Cell, HDF5Fluid, Boundary, CSVIteration, CSVCell]:
            cursor.execute(entity.get_schema())

        self.connection.commit()

    def execute_many(self, command: str, args: list):
        cursor = self.connection.cursor()

        cursor.executemany(command, args)

        self.connection.commit()

    def insert(self, command: str, args: tuple = ()) -> int:
        cursor = self.connection.cursor()

        cursor.execute(command, args)

        self.connection.commit()

        return int(cursor.lastrowid)

    def insert_many(self, command: str, args: list):
        cursor = self.connection.cursor()

        cursor.executemany(command, args)

        self.connection.commit()

    def exists(self, command, *args) -> (bool, Union[int, None]):
        cursor = self.connection.cursor()

        cursor.execute(command, args)

        result = cursor.fetchone()

        if result is None:
            return False, None
        else:
            return True, tuple(result)[0]

    def select_all(self, command, *args) -> list:
        return list(self.connection.execute(command, args))

    def select_prepared(self, command: str, args: list) -> list:
        return list(self.connection.execute(command, args))

    def select_one(self, command, *args) -> tuple or None:
        row = self.connection.execute(command, args).fetchone()

        if row is None:
            return None
        else:
            return tuple(row)

    def count(self, command, *args) -> int:
        return len(list(self.connection.execute(command, *args).fetchall()))

    def remove_simulation(self, simulation_name: str):
        """
        Remove a simulation from the database by its name
        :param simulation_name: Name of the simulation to remove
        """

        self.remove(f"DELETE from simulation WHERE simulation_name='{simulation_name}';")

    def cleanup_before_continue(self, simulation_id) -> (int, int):
        """
        Remove the config, boundary and last inserted HDF5 or CSV iteration since that might be incomplete.
        :param simulation_id: The id of the simulation to continue inserting
        :return: (hdf5 iteration to continue from, csv iteration to continue from)
        """

        self.remove(f"DELETE FROM config WHERE simulation_id={simulation_id};")
        self.remove(f"DELETE FROM boundary WHERE simulation_id={simulation_id};")

        (_, hdf5_iteration) = self.exists(f"""SELECT iteration FROM hdf5_iteration 
        WHERE simulation_id={simulation_id} 
        ORDER BY iteration DESC 
        LIMIT 1
        ;""")

        if hdf5_iteration is not None:
            (_, csv_iteration) = self.exists(f"""SELECT iteration FROM csv_iteration 
                    WHERE simulation_id={simulation_id} 
                    ORDER BY iteration DESC 
                    LIMIT 1
                    ;""")

            if csv_iteration is not None:
                # Remove last iteration since it might be incomplete due to a crash during last run
                self.remove(f"DELETE FROM csv_iteration WHERE simulation_id={simulation_id} AND iteration={csv_iteration};")
                return hdf5_iteration + 1, csv_iteration
            else:
                # Remove last iteration since it might be incomplete due to a crash during last run
                self.remove(f"DELETE FROM hdf5_iteration WHERE simulation_id={simulation_id} AND iteration={hdf5_iteration};")

                return hdf5_iteration, 0
        else:
            return 0, 0

    def remove(self, command):
        self.connection.execute("PRAGMA foreign_keys = ON")

        cursor = self.connection.cursor()

        cursor.execute(command)

        self.connection.commit()

    def start_progress_tracker(self, handler):
        """
        A progress tracker used to visualize the progress of a query.
        """
        self.connection.set_progress_handler(handler, 10)

    def stop_progress_tracker(self):
        self.connection.set_progress_handler(None, 10)

    def adapt_array(self, array: np.ndarray or None):
        """
        Save a numpy array in the matrix directory and save the filename in the database
        """

        if array is None:
            return None

        filename = f"{self.matrix_directory}/{uuid4()}"

        # Size of array in GB
        array_size = array.itemsize * array.size / 1e9

        # Max size of MATLAB array
        if array_size > 1.5:
            for i, matrix in enumerate(np.array_split(array, int(ceil(array_size / 1.5)))):
                savemat(f"{filename}_{i}.mat", {"array": matrix})
        else:
            savemat(f"{filename}.mat", {"array": array})

        return filename

    def convert_array(self, filename: str or None):
        """
        Load a numpy array from the matrix directory by the given filename
        """

        if filename is None:
            return None

        files = sorted([f for f in glob(f"{filename.decode()}*.mat")])

        if len(files) == 0:
            print(f"Could not find matrix at {filename}", file=sys.stderr)
            return None

        array: np.ndarray = loadmat(files[0]).get('array')
        for file in files[1:]:
            array = np.concatenate((array, loadmat(file).get('array')))

        return array

