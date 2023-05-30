"""

"""

from __future__ import annotations
from dataclasses import dataclass
import struct
from contextlib import ExitStack, contextmanager
from typing import NewType
from os import PathLike

import numpy as np
from scipy.interpolate import (
    RegularGridInterpolator as BaseInterpolator,
)
from scipy.integrate import (
    RK45 as BaseIntegrator,
)

from .utils import FileLike

__all__ = [
    'Interpolator', 'Integrator',
]


# SEC = 0
PRS = 0
LNG = 1
LAT = 2

Seconds = NewType('Seconds', float)
Pressure = NewType('Pressure', float)
Longitude = NewType('Longitude', float)
Latitude = NewType('Latitude', float)


def _calculate_ghost_wraparound(points: Tuple[float, ...], *, lo, hi) -> Tuple[float, float]:
    """Calculate the positions for the ghost regions, from simple wraparound.

    >>> _calculate_ghost_wraparound((0.0, 357.5), lo=0.0, hi=360.0)
    (-2.5, 360.0)
    >>> _calculate_ghost_wraparound((-88.7, 88.7), lo=-90.0, hi=90.0)
    (-91.3, 91.3)

    """
    LO, HI = lo, hi
    lo, hi = points[0], points[-1]

    # Assume: LO=0 HI=360 lo=0.0 hi=357.5
    # We want the newly added lo to be 0.0-((360-357.5)+(0-0.0)) lo-((HI-hi)+(LO-lo))
    # We want the newly added hi to be 357.5+(357.5-360)+(0.0-0)

    lo, hi = LO - (HI - hi), HI - (LO - lo)

    return lo, hi


@dataclass
class Interpolator:
    multiplier: Optional[float]
    points: Tuple[np.ndarray, ...]
    values: np.ndarray
    interpolator: BaseInterpolator

    @classmethod
    def from_file(cls, fileobj: FileLike, *, multiplier: Optional[float]=None) -> Data:
        with ExitStack() as stack:
            if isinstance(fileobj, str) or isinstance(fileobj, PathLike):
                fileobj = stack.enter_context(open(fileobj, 'rb'))

            def read(format):
                return struct.unpack(format, fileobj.read(struct.calcsize(format)))

            name, = read('4s')
            assert name in (b'UGRD', b'VGRD', b'VVEL'), \
                f'Unexpected name: {name!r}'

            dimensions, = read('I')
            assert dimensions == 4

            sec_count, = read('I')
            sec_points = read(sec_count * 'f')

            prs_count, = read('I')
            prs_points = read(prs_count * 'f')

            lng_count, = read('I')
            lng_points = read(lng_count * 'f')

            lat_count, = read('I')
            lat_points = read(lat_count * 'f')

            lng_point_lo, lng_point_hi = _calculate_ghost_wraparound(lng_points, lo=0.0, hi=360.0)
            lat_point_lo, lat_point_hi = _calculate_ghost_wraparound(lat_points, lo=-90.0, hi=90.0)

            values = np.zeros((sec_count, prs_count, 1+lng_count+1, 1+lat_count+1), dtype=np.float32, order='C')

            lng_points = (lng_point_lo,) + lng_points + (lng_point_hi,)
            lat_points = (lat_point_lo,) + lat_points + (lat_point_hi,)

            points = (sec_points, prs_points, lng_points, lat_points)

            shape = (sec_count, prs_count, lng_count, lat_count)
            chunks, chunk_shape = shape[0], shape[1:]
            for chunk in range(chunks):
                values[chunk, :, 1:-1, 1:-1] = np.fromfile(fileobj, dtype=np.float32, count=np.prod(chunk_shape)).reshape(chunk_shape)

            values[:, :, 0-(0), :] = values[:, :, -2, :]
            values[:, :, 0-(1), :] = values[:, :, +1, :]
            values[:, :, :, 0-(0)] = values[:, :, :, -2]
            values[:, :, :, 0-(1)] = values[:, :, :, +1]

#             values = np.memmap(fileobj, dtype=np.float32, mode='r', offset=offset, shape=shape, order='C')

            interpolator = BaseInterpolator(
                points=points,
                values=values,
                method='linear',
                bounds_error=False,
                fill_value=None,
            )

            return cls(
                multiplier=multiplier,
                points=points,
                values=values,
                interpolator=interpolator,
            )

    def __call__(
        self,
        t: Seconds,
        x: Tuple[Pressure, Longitude, Latitude],
    ) -> float:
        x = np.vstack((t, x)).T
        y = self.interpolator(x)
        if self.multiplier is not None:
            y *= self.multiplier
        return y


@dataclass
class Integrator:
    ugrd: Interpolator
    vgrd: Interpolator
    vvel: Interpolator

    @classmethod
    def from_files(cls, *, ugrd: FileLike, vgrd: FileLike, vvel: FileLike) -> Climate:
        ugrd = Interpolator.from_file(ugrd)
        vgrd = Interpolator.from_file(vgrd)
        vvel = Interpolator.from_file(vvel)

        return cls(
            ugrd=ugrd,
            vgrd=vgrd,
            vvel=vvel,
        )

    def __call__(self, t: Sec, x: Tuple[Prs, Lng, Lat]) -> Tuple[Val, Val, Val]:
        ugrd: Val = self.ugrd(t, x)
        vgrd: Val = self.vgrd(t, x)
        vvel: Val = self.vvel(t, x)
        
        return (ugrd, vgrd, vvel)

    def integrate(self,
        *,
        t0: Sec,
        y0: Tuple[Prs, Lng, Lat],
        tf: Sec,
    ) -> Iterator[Tuple[Sec, Tuple[Prs, Lng, Lat]]]:
        _1_DAY_IN_SECONDS = 60 * 60 * 24

        y0 = np.array(y0)

        integrator = BaseIntegrator(
            fun=self,
            t0=t0,
            y0=y0,
            t_bound=tf,
            max_step=_1_DAY_IN_SECONDS,
            vectorized=True,
        )

        while True:
            message = integrator.step()
            if message is not None:
                print(f'{message = }')

            t = integrator.t
            y = integrator.y

            if y[PRS] < 0: break
            y[y[LNG] < 0.0] += 360.0
            y[y[LNG] > 360.0] -= 360.0
            y[y[LAT] < -90.0] += 180.0
            y[y[LAT] > 90.0] -= 180.0

            yield t, y

            if integrator.status != 'running':
                break