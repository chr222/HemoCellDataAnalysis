from math import floor
from statistics import mean
from time import time
from dataclasses import dataclass


class ProgressFunction:
    def __init__(self, total_iterations: int, message_prefix: str = ''):
        self.total_iterations = total_iterations
        self.description = message_prefix
        self.iterations_times = []

        print(f"\r{self.description} | Completed 0/{self.total_iterations} iterations", end="")

    def run(self, function: any, *args):
        """
        Run iteration
        :param function: Function to run
        :param args: Arguments of function
        :return: Result of the function
        """

        start_time = time()
        result = function(*args)
        self.iterations_times.append(time() - start_time)

        self.print_progress()

        return result

    @property
    def time_remaining(self) -> float:
        """
        Estimated time need to run the remaining iterations
        """

        return mean(self.iterations_times[-100:]) * (self.total_iterations - len(self.iterations_times))

    def print_progress(self):
        print(f"\r{self.description} | Completed {len(self.iterations_times)}/{self.total_iterations} iterations | Time remaining: {time_remaining_to_string(self.time_remaining)}", end="")


@dataclass
class StatusHandler:
    """
    Writes the status of the program by overwriting lines
    """

    prefix: str = ''
    last_line: str = ''

    def print(self, message: str):
        """
        Overwrite last line
        """

        line = self.prefix + message
        print(f'\r{line: <{len(self.last_line)}}', end="")

        if self.prefix == '':
            self.prefix = message

        self.last_line = line

    def println(self, message: str):
        """
        Last message of batch and continue on new line
        """

        print(f'\r{message: <{len(self.last_line)}}')
        self.prefix = ''
        self.last_line = ''


def time_remaining_to_string(remaining_seconds: float) -> str:
    hours = int(floor(remaining_seconds / 3600))
    minutes = int(floor((remaining_seconds % 3600) / 60))
    seconds = int(remaining_seconds % 60)

    if remaining_seconds > 3600:
        return f"{hours}h{minutes}m{seconds}s"
    elif remaining_seconds > 60:
        return f"{minutes}m{seconds}s"
    else:
        return f"{seconds}s"
