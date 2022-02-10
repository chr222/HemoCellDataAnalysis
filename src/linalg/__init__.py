from dataclasses import dataclass
from math import sqrt
import numpy as np


@dataclass
class Vector3Int:
    x: int
    y: int
    z: int


@dataclass
class Vector3:
    x: float
    y: float
    z: float

    @property
    def magnitude(self) -> float:
        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __array__(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def __getitem__(self, item: str):
        return getattr(self, item)

    def __mul__(self, other):
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        return Vector3(self.x / other, self.y / other, self.z / other)


@dataclass
class Tensor9:
    xx: float
    xy: float
    xz: float
    yx: float
    yy: float
    yz: float
    zx: float
    zy: float
    zz: float

    def __array__(self) -> np.ndarray:
        return np.array([self.xx, self.xy, self.xz, self.yx, self.yy, self.yz, self.zx, self.zy, self.zz])

    @property
    def x_plane(self) -> float:
        return sqrt(pow(self.xy, 2) + pow(self.xz, 2))

    @property
    def y_plane(self) -> float:
        return sqrt(pow(self.yx, 2) + pow(self.yz, 2))

    @property
    def z_plane(self) -> float:
        return sqrt(pow(self.zx, 2) + pow(self.zy, 2))

    @property
    def x_face(self) -> float:
        return sqrt(pow(self.xy, 2) + pow(self.xz, 2) + pow(self.xx, 2))

    @property
    def y_face(self) -> float:
        return sqrt(pow(self.yx, 2) + pow(self.yz, 2) + pow(self.yy, 2))

    @property
    def z_face(self) -> float:
        return sqrt(pow(self.zx, 2) + pow(self.zy, 2) + pow(self.zz, 2))

    @property
    def magnitude(self) -> float:
        return sqrt(np.sum(np.power(np.array(self), 2)))

    def __getitem__(self, item: str):
        return getattr(self, item)


@dataclass
class Tensor6:
    xx: float
    xy: float
    xz: float
    yy: float
    yz: float
    zz: float

    def __array__(self) -> np.ndarray:
        return np.array([self.xx, self.xy, self.xz, self.yy, self.yz, self.zz])

    def tensor_9_array(self) -> np.ndarray:
        return np.array([self.xx, self.xy, self.xz, self.xy, self.yy, self.yz, self.xz, self.yz, self.zz])

    @property
    def x_plane(self) -> float:
        return sqrt(pow(self.xy, 2) + pow(self.xz, 2))

    @property
    def y_plane(self) -> float:
        return sqrt(pow(self.xy, 2) + pow(self.yz, 2))

    @property
    def z_plane(self) -> float:
        return sqrt(pow(self.xz, 2) + pow(self.yz, 2))

    @property
    def x_face(self) -> float:
        return sqrt(pow(self.xy, 2) + pow(self.xz, 2) + pow(self.xx, 2))

    @property
    def y_face(self) -> float:
        return sqrt(pow(self.xy, 2) + pow(self.yz, 2) + pow(self.yy, 2))

    @property
    def z_face(self) -> float:
        return sqrt(pow(self.xz, 2) + pow(self.yz, 2) + pow(self.zz, 2))

    @property
    def magnitude(self) -> float:
        return sqrt(np.sum(np.power(self.tensor_9_array(), 2)))

    def __getitem__(self, item: str):
        return getattr(self, item)
