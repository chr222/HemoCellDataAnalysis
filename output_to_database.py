from src.sql.connection import Connection
from src.sql.entity.simulation import parse_simulation
from src.progress import time_remaining_to_string
import params

from argparse import ArgumentParser
import sys
from time import time


def import_simulation(simulation_name: str, data_directory: str, config_path: str = None):
    """
    Import the simulation into the database
    :param simulation_name: The name you want to give the simulation in the database
    :param data_directory: The directory with data that needs to be imported
    :param config_path: The path to the config file. By default it will use the one in the output directory
    """

    try:
        # Setup connection with the database and create it if it does not exist yet
        connection = Connection(database_name=params.DATABASE_NAME)

        # Parse the simulation
        start_time = time()
        parse_simulation(
            connection,
            simulation_name,
            data_directory,
            config_path
        )
        sys.stdout.write(f"\rImported simulation \"{simulation_name}\" in {time_remaining_to_string(time() - start_time)} total\n\n")
        sys.stdout.flush()

    # Handle pausing the import process
    except KeyboardInterrupt:
        print('\n\nEnding import process, you can continue this later.\n\n')


def setup_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Parse the output data directory of a HemoCell simulation and write it to a database.')

    parser.add_argument('name', metavar='name', type=str, help='The name you want to give to the simulation in the database.')
    parser.add_argument('directory', metavar='data_directory', type=str, help='The path to the output directory of the HemoCell simulation.')
    parser.add_argument('--config', type=str, default=None, help='(OPTIONAL) The path to the config file (otherwise it will use the one in the output directory)')

    return parser


if __name__ == '__main__':
    arg_parser = setup_argument_parser()
    args = arg_parser.parse_args()

    import_simulation(args.name, args.directory, args.config)
