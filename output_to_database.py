from src.sql.connection import Connection
from src.sql.entity.simulation import parse_simulation
from src.progress import time_remaining_to_string
import params

from argparse import ArgumentParser
import sys
from time import time


def main(simulation_name: str, data_directory: str):
    try:
        connection = Connection(database_name=params.DATABASE_NAME)

        start_time = time()
        parse_simulation(
            connection,
            simulation_name,
            data_directory
        )
        sys.stdout.write(f"\rInserted simulation \"{simulation_name}\" in {time_remaining_to_string(time() - start_time)} total\n\n")
        sys.stdout.flush()
    except KeyboardInterrupt:
        print('\n\nEnding insertion process, you can continue this later.\n\n')


def setup_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Parse the output data directory of a HemoCell simulation and write it to a database.')

    parser.add_argument('name', metavar='name', type=str, help='The name you want to give to the simulation in the database.')
    parser.add_argument('directory', metavar='data_directory', type=str, help='The path to the output directory of the HemoCell simulation.')

    return parser


if __name__ == '__main__':
    arg_parser = setup_argument_parser()
    args = arg_parser.parse_args()

    main(args.name, args.directory)
