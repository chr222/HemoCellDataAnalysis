import inspect
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import cm
from pathlib import Path
from matplotlib import ticker
import numpy as np
from numpy import nan
from scipy.signal import convolve2d
from typing import Union
import warnings
import numpy.ma as ma
import seaborn as sns


def nanmean(x: Union[np.ndarray, list], *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, *args, **kwargs)


def nanmedian(x: Union[np.ndarray, list], *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(x, *args, **kwargs)


def nanmax(x: Union[np.ndarray, list], *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(x, *args, **kwargs)


def nanmin(x: Union[np.ndarray, list], *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmin(x, *args, **kwargs)


def fmax(x: Union[np.ndarray, list], *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.fmax(x, *args, **kwargs)


def fmin(x: Union[np.ndarray, list], *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.fmin(x, *args, **kwargs)


class Graphics:
    def __init__(self, output_directory: Path, save_figures: bool):
        self.output_directory = output_directory
        self.save_figures = save_figures
        self.x_ticks = None

        self.stat_to_axis = {
            'min': 'Min',
            'max': 'Max',
            'mean': 'Mean',
            'median': 'Median',
            'mean_std': 'Mean',
            'mean_minmax': 'Mean',
            'median_std': 'Median',
            'median_minmax': 'Median'
        }

        self.data_type_unit = {
            'velocity': '$mm/s$', #\mu
            'velocity-y': '$\mu m/s$',
            'shear stress': '$Pa$',
            'shear rate': '$s^{-1}$',
            'elongation rate': '$s^{-1}$',
            'height': '$\mu m$',
            'distance from center': '$\mu m$',
            'horizontal movement': '$\mu m$',
            'vertical movement': '$\mu m$',
            'hematocrit': '%'
        }

        if not self.output_directory.is_dir():
            self.output_directory.mkdir()

    def set_x_ticks(self, x_ticks: np.ndarray):
        self.x_ticks = x_ticks

    @staticmethod
    def pick_ticks(ticks: np.ndarray, n: int) -> (np.ndarray, np.ndarray):
        indices = np.linspace(0, len(ticks) - 1, n, dtype=int)

        return indices, ticks[indices]

    @staticmethod
    def distance_from_radius_map(x_len: int, z_len: int) -> dict:
        mesh = np.mgrid[:x_len, :z_len]
        distances = np.array(
            np.round(
                np.sqrt(np.power(mesh[0] - (x_len / 2 - 0.5), 2) + np.power(mesh[1] - (z_len / 2 - 0.5), 2))
            ),
            dtype=int
        )

        return {distance: np.argwhere(distances == distance) for distance in np.unique(distances)}

    @staticmethod
    def distance_from_radius_map_1d(diameter: int) -> list:
        mesh = np.mgrid[:diameter, :diameter]

        middle = diameter // 2
        distances = np.array(np.round(np.sqrt(np.power(mesh[0] - middle, 2) + np.power(mesh[1] - middle, 2))), dtype=int)

        left = distances[middle, :middle]
        right = distances[middle, middle:]

        return [np.argwhere(distances[:, :middle] == distance) for distance in left] + \
               [np.argwhere(distances[:, middle:] == distance) + [0, middle] for distance in right]


    def plot(self, title: str = None, x_label: str = None, y_label: str = None, figure_name: str or None = None):
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()

        if self.save_figures:
            if figure_name is None:
                # Figure gets name of parent function
                figure_name = inspect.stack()[1].function

            plt.savefig(self.output_directory / f"{figure_name}.png")
        else:
            plt.show()

        plt.close('all')

    def try_plot(self, prefix: str or None, data_type: str or None, stat: str, x_label: str = 'Time (ms)', y_label: str or None = None, label: str or None = None, suffix: str = '_over_time'):
        if prefix is not None and data_type is not None:
            if y_label is None:
                y_label = f'{self.stat_to_axis[stat]} {data_type} ({self.data_type_unit[data_type]})'

            if label is not None:
                plt.legend()

            self.plot(
                None,
                x_label,
                y_label,
                figure_name=f'{prefix}_{stat}_{data_type.replace("-", "_").replace(" ", "_")}{suffix}'
            )

    def plot_max_over_time(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None):
        plt.plot(self.x_ticks[:len(data)], nanmax(data, axis=(1, 2, 3)))
        self.try_plot(prefix, data_type, 'max')

    def plot_mean_over_time(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None):
        plt.plot(self.x_ticks[:len(data)], nanmean(data, axis=(1, 2, 3)))
        self.try_plot(prefix, data_type, 'mean')

    def plot_median_over_time(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None):
        plt.plot(self.x_ticks[:len(data)], nanmedian(data, axis=(1, 2, 3)))
        self.try_plot(prefix, data_type, 'median')

    def plot_mean_std_over_time(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None):
        mean = nanmean(data, axis=(1, 2, 3))
        std = np.nanstd(data, axis=(1, 2, 3))
        lower = fmax(mean - std, nanmin(data, axis=(1, 2, 3)))
        upper = fmin(mean + std, nanmax(data, axis=(1, 2, 3)))

        x = self.x_ticks[:len(data)]

        plt.plot(x, mean)
        plt.fill_between(x, lower, upper, alpha=0.5)
        self.try_plot(prefix, data_type, 'mean_std')

    def plot_mean_minmax_over_time(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None):
        mean = nanmean(data, axis=(1, 2, 3))
        lower = nanmin(data, axis=(1, 2, 3))
        upper = nanmax(data, axis=(1, 2, 3))

        x = self.x_ticks[:len(data)]

        plt.plot(x, mean)
        plt.fill_between(x, lower, upper, alpha=0.5)
        self.try_plot(prefix, data_type, 'mean_minmax')

    def plot_median_std_over_time(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None):
        median = nanmedian(data, axis=(1, 2, 3))
        std = np.nanstd(data, axis=(1, 2, 3))
        lower = fmax(median - std, nanmin(data, axis=(1, 2, 3)))
        upper = fmin(median + std, nanmax(data, axis=(1, 2, 3)))

        x = self.x_ticks[:len(data)]

        plt.plot(x, median)
        plt.fill_between(x, lower, upper, alpha=0.5)
        self.try_plot(prefix, data_type, 'median_std')

    def plot_median_minmax_over_time(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None):
        median = nanmedian(data, axis=(1, 2, 3))
        lower = nanmin(data, axis=(1, 2, 3))
        upper = nanmax(data, axis=(1, 2, 3))

        x = self.x_ticks[:len(data)]

        plt.plot(x, median)
        plt.fill_between(x, lower, upper, alpha=0.5)
        self.try_plot(prefix, data_type, 'median_minmax')

    def plot_mean_over_height_over_time(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None):
        mean = nanmean(data, axis=(2, 3))

        plt.imshow(np.transpose(mean), cmap=cm.winter, origin='lower')
        plt.colorbar(
            cm.ScalarMappable(cmap='winter'),
            format=ticker.FuncFormatter(lambda value, _: f"{(value * nanmax(mean)):.2f}"),
            label=f'Mean {data_type} ({self.data_type_unit[data_type]})'
        )
        plt.xticks(*self.pick_ticks(self.x_ticks[:len(data)], 5))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / 2}'))
        self.try_plot(prefix, data_type, 'mean_over_height', y_label='Height ($\mu m$)')

    def plot_median_over_height_over_time(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None):
        median = nanmedian(data, axis=(2, 3))

        plt.imshow(np.transpose(median), cmap=cm.winter, origin='lower')
        plt.colorbar(
            cm.ScalarMappable(cmap='winter'),
            format=ticker.FuncFormatter(lambda value, _: f"{(value * nanmax(median)):.2f}"),
            label=f'Median {data_type} ({self.data_type_unit[data_type]})'
        )
        plt.xticks(*self.pick_ticks(self.x_ticks[:len(data)], 5))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / 2}'))

        self.try_plot(prefix, data_type, 'median_over_height', y_label='Height ($\mu m$)')

    def plot_max_over_height_over_time(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None):
        _max = nanmax(data, axis=(2, 3))

        plt.imshow(np.transpose(_max), cmap=cm.winter, origin='lower')
        plt.colorbar(
            cm.ScalarMappable(cmap='winter'),
            format=ticker.FuncFormatter(lambda value, _: f"{(value * nanmax(_max)):.2f}"),
            label=f'Max {data_type} ({self.data_type_unit[data_type]})'
        )
        plt.xticks(*self.pick_ticks(self.x_ticks[:len(data)], 5))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / 2}'))

        self.try_plot(prefix, data_type, 'max_over_height', y_label='Height ($\mu m$)')

    def plot_mean_over_radius_over_time(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None):
        distance_map = self.distance_from_radius_map(len(data[0][1]), len(data[0][2]))

        mean = np.array([
            [nanmean(data_2d[coordinates]) for distance, coordinates in distance_map.items()]
            for data_2d in nanmean(data, axis=1)
        ])

        plt.imshow(np.transpose(mean), cmap=cm.winter, origin='lower')
        plt.colorbar(
            cm.ScalarMappable(cmap='winter'),
            format=ticker.FuncFormatter(lambda value, _: f"{(value * nanmax(mean)):.2f}"),
            label=f'Mean {data_type} ({self.data_type_unit[data_type]})'
        )
        plt.xticks(*self.pick_ticks(self.x_ticks[:len(data)], 5))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / 2}'))

        self.try_plot(prefix, data_type, 'mean_over_radius', y_label='Distance from center ($\mu m$)')

    def plot_median_over_radius_over_time(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None):
        distance_map = self.distance_from_radius_map(len(data[0][1]), len(data[0][2]))

        median = np.array([
            [nanmedian(data_2d[coordinates]) for distance, coordinates in distance_map.items()]
            for data_2d in nanmedian(data, axis=1)
        ])

        plt.imshow(np.transpose(median), cmap=cm.winter, origin='lower')
        plt.colorbar(
            cm.ScalarMappable(cmap='winter'),
            format=ticker.FuncFormatter(lambda value, _: f"{(value * nanmax(median)):.2f}"),
            label=f'Median {data_type} ({self.data_type_unit[data_type]})'
        )
        plt.xticks(*self.pick_ticks(self.x_ticks[:len(data)], 5))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / 2}'))

        self.try_plot(prefix, data_type, 'median_over_radius', y_label='Distance from center ($\mu m$)')

    def plot_max_over_radius_over_time(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None):
        distance_map = self.distance_from_radius_map(len(data[0][1]), len(data[0][2]))

        _max = np.array([
            [nanmax(data_2d[coordinates]) for distance, coordinates in distance_map.items()]
            for data_2d in nanmax(data, axis=1)
        ])

        plt.imshow(np.transpose(_max), cmap=cm.winter, origin='lower')
        plt.colorbar(
            cm.ScalarMappable(cmap='winter'),
            format=ticker.FuncFormatter(lambda value, _: f"{(value * nanmax(_max)):.2f}"),
            label=f'Max {data_type} ({self.data_type_unit[data_type]})'
        )
        plt.xticks(*self.pick_ticks(self.x_ticks[:len(data)], 5))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y / 2}'))

        self.try_plot(prefix, data_type, 'max_over_radius', y_label='Distance from center ($\mu m$)')

    def csv_plot_mean_over_time(self, data: list, prefix: str or None = None, data_type: str or None = None, color=None, label=None):
        plt.plot(self.x_ticks[:len(data)], np.array([np.mean(i) for i in data]), color=color, label=label)
        self.try_plot(prefix, data_type, 'mean', label=label)

    def csv_plot_median_over_time(self, data: list, prefix: str or None = None, data_type: str or None = None, color=None, label=None):
        plt.plot(self.x_ticks[:len(data)], np.array([np.median(i) for i in data]), color=color, label=label)
        self.try_plot(prefix, data_type, 'median', label=label)

    def csv_plot_max_over_time(self, data: list, prefix: str or None = None, data_type: str or None = None, color=None, label=None):
        plt.plot(self.x_ticks[:len(data)], np.array([np.max(i) for i in data]), color=color, label=label)
        self.try_plot(prefix, data_type, 'max', label=label)

    def csv_plot_mean_std_over_time(self, data: list, prefix: str or None = None, data_type: str or None = None, color=None, label=None):
        mean = np.array([np.mean(i) for i in data])
        std = np.array([np.std(i) for i in data])
        lower = fmax(mean - std, np.array([np.min(i) for i in data]))
        upper = fmin(mean + std, np.array([np.max(i) for i in data]))

        x = self.x_ticks[:len(data)]

        plt.plot(x, mean, color=color, label=label)
        plt.fill_between(x, lower, upper, color=color, alpha=0.5)
        self.try_plot(prefix, data_type, 'mean_std', label=label)

    def csv_plot_mean_minmax_over_time(self, data: list, prefix: str or None = None, data_type: str or None = None, color=None, label=None):
        mean = np.array([np.mean(i) for i in data])
        lower = np.array([np.min(i) for i in data])
        upper = np.array([np.max(i) for i in data])

        x = self.x_ticks[:len(data)]

        plt.plot(x, mean, color=color, label=label)
        plt.fill_between(x, lower, upper, color=color, alpha=0.5)
        self.try_plot(prefix, data_type, 'mean_minmax', label=label)

    def csv_plot_median_std_over_time(self, data: list, prefix: str or None = None, data_type: str or None = None, color=None, label=None):
        median = np.array([np.median(i) for i in data])
        std = np.array([np.std(i) for i in data])
        lower = fmax(median - std, np.array([np.min(i) for i in data]))
        upper = fmin(median + std, np.array([np.max(i) for i in data]))

        x = self.x_ticks[:len(data)]

        plt.plot(x, median, color=color, label=label)
        plt.fill_between(x, lower, upper, color=color, alpha=0.5)
        self.try_plot(prefix, data_type, 'median_std', label=label)

    def csv_plot_median_minmax_over_time(self, data: list, prefix: str or None = None, data_type: str or None = None, color=None, label=None):
        median = np.array([np.median(i) for i in data])
        lower = np.array([np.min(i) for i in data])
        upper = np.array([np.max(i) for i in data])

        x = self.x_ticks[:len(data)]

        plt.plot(x, median, color=color, label=label)
        plt.fill_between(x, lower, upper, color=color, alpha=0.5)
        self.try_plot(prefix, data_type, 'median_minmax', label=label)

    def plot_over_time(self, y: np.ndarray, prefix: str or None = None, data_type: str or None = None, color=None, label=None):
        plt.plot(self.x_ticks[:len(y)], y, color=color, label=label)
        self.try_plot(prefix, data_type, 'count', y_label='Number of cells', label=label)

    def plot_moving_average_over_time(self, data: np.ndarray, window: int, prefix: str or None = None, data_type: str or None = None):
        if window % 2 == 0:
            raise Exception(f'Moving average window should be uneven, but is even: {window}')

        plt.plot(
            self.x_ticks[:len(data)],
            np.convolve(
                np.pad(nanmean(data, axis=(1, 2, 3)), pad_width=window // 2, mode='edge'),
                np.ones(window),
                mode='valid'
            ) / window
        )
        self.try_plot(
            prefix,
            data_type,
            stat=f'moving_average_{window}',
            y_label=f'Moving average (n={window}) of {data_type} ({self.data_type_unit[data_type]})'
        )

    def csv_plot_moving_average_over_time(self, data: list, window: int, prefix: str or None = None, data_type: str or None = None, color=None, label=None):
        if window % 2 == 0:
            raise Exception(f'Moving average window should be uneven, but is even: {window}')

        plt.plot(
            self.x_ticks[:len(data)],
            np.convolve(
                np.pad(np.array([np.mean(i) for i in data]), pad_width=window // 2, mode='edge'),
                np.ones(window),
                mode='valid'
            ) / window,
            color=color,
            label=label
        )
        self.try_plot(
            prefix,
            data_type,
            stat=f'moving_average_{window}',
            y_label=f'Moving average (n={window}) of {data_type} ({self.data_type_unit[data_type]})' if data_type is not None else None,
            label=label
        )

    def plot_vertical_slice(self, data: np.ndarray, prefix: str, data_type: str):
        distances = self.distance_from_radius_map_1d(data.shape[1])
        mapped_data = []
        for y in range(data.shape[0]):
            mapped_data.append([nanmean(data[(np.array([y] * coords.shape[0]), coords[:, 0], coords[:, 1])]) for coords in distances])
        mapped_data = np.array(mapped_data)

        window = 7
        moving_average = convolve2d(
            mapped_data,
            np.ones((window, window)),
            mode='valid'
        ) / (window * window)
        plt.imshow(moving_average, origin='lower')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 2}'))
        plt.colorbar(
            cm.ScalarMappable(),
            format=ticker.FuncFormatter(lambda value, _: f"{(value * nanmax(moving_average)):.2f}"),
            label=f'Mean {data_type} ({self.data_type_unit[data_type]})'
        )

        self.try_plot(
            prefix,
            data_type,
            stat=f'vertical_slice',
            x_label='Width ($\mu m$)',
            y_label='Height ($\mu m$)',
            suffix=''
        )

    def plot_horizontal_slice(self, data: np.ndarray, prefix: str, data_type: str):
        window = 7
        moving_average = convolve2d(
            nanmean(data, axis=0),
            np.ones((window, window)),
            mode='valid'
        ) / (window * window)
        plt.imshow(moving_average, origin='lower')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 2}'))
        plt.colorbar(
            cm.ScalarMappable(),
            format=ticker.FuncFormatter(lambda value, _: f"{(value * nanmax(moving_average)):.2f}"),
            label=f'Mean {data_type} ({self.data_type_unit[data_type]})'
        )

        self.try_plot(
            prefix,
            data_type,
            stat=f'horizontal_slice',
            x_label='x ($\mu m$)',
            y_label='z ($\mu m$)',
            suffix=''
        )

    def plot_moving_average_over_height(self, data: np.ndarray, window: int, prefix: str, data_type: str):
        if window % 2 == 0:
            raise Exception(f'Moving average window should be uneven, but is even: {window}')

        plt.plot(
            np.arange(len(data)),
            np.convolve(
                np.pad(nanmean(data, axis=(1, 2)), pad_width=window // 2, mode='edge'),
                np.ones(window),
                mode='valid'
            ) / window
        )
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 2}'))

        self.try_plot(
            prefix,
            data_type,
            stat=f'moving_average_{window}',
            y_label=f'Moving average (n={window}) of {data_type} ({self.data_type_unit[data_type]})',
            x_label='Height ($\mu m$)',
            suffix='_over_height'
        )

    def plot_moving_average_over_width(self, data: np.ndarray, window: int, prefix: str, data_type: str):
        if window % 2 == 0:
            raise Exception(f'Moving average window should be uneven, but is even: {window}')

        plt.plot(
            np.arange(data.shape[2]),
            np.convolve(
                np.pad(nanmean(data, axis=(0, 1)), pad_width=window // 2, mode='edge'),
                np.ones(window),
                mode='valid'
            ) / window
        )
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 2}'))

        self.try_plot(
            prefix,
            data_type,
            stat=f'moving_average_{window}',
            y_label=f'Moving average (n={window}) of {data_type} ({self.data_type_unit[data_type]})',
            x_label='Width ($\mu m$)',
            suffix='_over_width'
        )

    def plot_time_averaged_cross_section(self, data: np.ndarray, prefix: str or None = None, data_type: str or None = None, v_max: int or None=None):

        x_ax = np.linspace(0, round(len(data)/2), len(data)+1)
        y_ax = np.linspace(0, round(len(data[0])/2), len(data[0])+1)

        Zm = ma.masked_invalid(data)
        x_ax, y_ax = np.meshgrid(x_ax, y_ax)

        plt.figure(figsize=(11, 6))
        if 'v_max' in locals():
            im = plt.pcolormesh(x_ax, y_ax, Zm.T, cmap=plt.get_cmap('jet'), vmax=v_max)
        else:
            im = plt.pcolormesh(x_ax, y_ax, Zm.T, cmap=plt.get_cmap('jet'))
        cbar = plt.colorbar(im, cmap=plt.get_cmap('jet'))  # YlGn; magma_r viridis Spectral coolwarm
        cbar.ax.set_ylabel(ylabel=f'Time averaged'
                                  f' {data_type} ({self.data_type_unit[data_type]})', rotation=-90,
                           va="bottom")

        self.try_plot(
            prefix,
            data_type,
            stat='time-averaged',
            x_label='Length ($\mu m$)',
            #y_label=f'Time average {data_type} ({self.data_type_unit[data_type]})',
            y_label='Height ($\mu m$)',
            suffix='cross-section'
        )
