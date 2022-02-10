from src.sql.entity.block import Block
from src.sql.entity.config import Config
from src.sql.entity.csv_cell import CSVCell
from src.sql.entity.csv_iteration import CSVIteration
from src.sql.entity.hdf5_cell import Hdf5Cell
from src.sql.entity.hdf5_fluid import Fluid, Boundary
from src.sql.entity.hdf5_iteration import Hdf5Iteration
from src.sql.entity.simulation import Simulation

import io
import numpy as np
from os.path import isfile
import re
import sqlite3
from typing import Union


class Connection:
    connection: sqlite3.dbapi2

    def __init__(self, database_name: str):
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        sqlite3.register_converter("array", self.convert_array)

        db_exists = self.database_exists(database_name)

        self.connection = sqlite3.connect(database_name, timeout=3600, detect_types=sqlite3.PARSE_DECLTYPES)

        if not db_exists:
            print("Database does not exist yet. Creating schema...")
            self.create_schema()

    @staticmethod
    def database_exists(database_name: str) -> bool:
        return isfile(database_name)

    def create_schema(self) -> sqlite3.dbapi2:
        cursor = self.connection.cursor()

        for entity in [Simulation, Config, Block, Hdf5Iteration, Hdf5Cell, Fluid, Boundary, CSVIteration, CSVCell]:
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

    def exists(self, command) -> (bool, Union[int, None]):
        cursor = self.connection.cursor()

        cursor.execute(command)

        result = cursor.fetchone()

        if result is None:
            return False, None
        else:
            return True, tuple(result)[0]

    def select_all(self, command) -> list:
        return list(self.connection.execute(command))

    def select_prepared(self, command: str, args: list) -> list:
        return list(self.connection.execute(command, args))

    def select_one(self, command) -> tuple:
        cursor = self.connection.cursor()

        cursor.execute(command)

        return tuple(cursor.fetchone())

    def count(self, command) -> int:
        cursor = self.connection.cursor()

        cursor.execute(command)

        return len(list(cursor.fetchall()))

    def remove_simulation(self, simulation_name: str):
        self.connection.execute("PRAGMA foreign_keys = ON")

        cursor = self.connection.cursor()

        cursor.execute(f"DELETE from simulation WHERE simulation_name='{simulation_name}';")

        self.connection.commit()

    def cleanup_before_continue(self, simulation_id) -> (int, int):
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

    @staticmethod
    def find_json_string(prefix: str, data: str) -> str:
        possible_data = (re.search(prefix + ": ({.*})", data.replace("\t", " ").replace("\n", ""))).group(1)

        result = ""
        depth = 0
        for char in possible_data:
            result += char

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

                if depth == 0:
                    return result

        raise Exception("Could not find end of json_string")

    def start_progress_tracker(self, handler):
        self.connection.set_progress_handler(handler, 10)

    def stop_progress_tracker(self):
        self.connection.set_progress_handler(None, 10)

    @staticmethod
    def adapt_array(array):
        """
        http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
        """
        out = io.BytesIO()
        np.save(out, array)
        out.seek(0)
        return sqlite3.Binary(out.read())

    @staticmethod
    def convert_array(text):
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

