"""
Cubic Spline interpolation using scipy.interpolate
Provides utilities for position, curvature, yaw, and arclength calculation
"""

import math

import numpy as np
import scipy.optimize as so
from scipy import interpolate
from typing import Union, Optional

from numba import njit
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point: np.ndarray, trajectory: np.ndarray) -> tuple:
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    Parameters
    ----------
    point: np.ndarray
        The 2d point to project onto the trajectory
    trajectory: np.ndarray
        The trajectory to project the point onto, shape (N, 2)
        The points must be unique. If they are not unique, a divide by 0 error will destroy the world

    Returns
    -------
    nearest_point: np.ndarray
        The nearest point on the trajectory
    distance: float
        The distance from the point to the nearest point on the trajectory
    t: float
    min_dist_segment: int
        The index of the nearest point on the trajectory
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return (
        projections[min_dist_segment],
        dists[min_dist_segment],
        t[min_dist_segment],
        min_dist_segment,
    )

class CubicSplineND:
    """
    Cubic CubicSplineND class.

    Attributes
    ----------
    s : list
        cumulative distance along the data points.
    xs : np.ndarray
        x coordinates for data points.
    ys : np.ndarray
        y coordinates for data points.
    spsi: np.ndarray
        yaw angles for data points.
    ks : np.ndarray
        curvature for data points.
    vxs : np.ndarray
        velocity for data points.
    axs : np.ndarray
        acceleration for data points.
    """
    def __init__(self,
        xs: np.ndarray,
        ys: np.ndarray,
        spsi: Optional[np.ndarray] = None,
        ks: Optional[np.ndarray] = None,
        vxs: Optional[np.ndarray] = None,
        axs: Optional[np.ndarray] = None,
    ):
        self.xs = xs
        self.ys = ys
        self.spsi = spsi # Lets us know if yaw was provided
        self.ks = ks # Lets us know if curvature was provided
        self.vxs = vxs # Lets us know if velocity was provided
        self.axs = axs # Lets us know if acceleration was provided

        spsi_spline = spsi if spsi is not None else np.zeros_like(xs)
        ks_spline = ks if ks is not None else np.zeros_like(xs)
        vxs_spline = vxs if vxs is not None else np.zeros_like(xs)
        axs_spline = axs if axs is not None else np.zeros_like(xs)

        self.points = np.c_[self.xs, self.ys, spsi_spline, ks_spline, vxs_spline, axs_spline]
        if not np.all(self.points[-1] == self.points[0]):
            self.points = np.vstack(
                (self.points, self.points[0])
            )  # Ensure the path is closed
        self.s = self.__calc_s(self.points[:, 0], self.points[:, 1])
        # Use scipy CubicSpline to interpolate the points with periodic boundary conditions
        # This is necesaxsry to ensure the path is continuous
        self.spline = interpolate.CubicSpline(self.s, self.points, bc_type="periodic")

    def __calc_s(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calc cumulative distance.

        Parameters
        ----------
        x : list
            x coordinates for data points.
        y : list
            y coordinates for data points.

        Returns
        -------
        s : np.ndarray
            cumulative distance along the data points.
        """
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return np.array(s)

    def calc_position(self, s: float) -> np.ndarray:
        """
        Calc position at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        x : float | None
            x position for given s.
        y : float | None
            y position for given s.
        """
        x,y = self.spline(s)[:2]
        return x,y

    def calc_curvature(self, s: float) -> Optional[float]:
        """
        Calc curvature at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        k : float
            curvature for given s.
        """
        if self.ks is None: # curvature was not provided => numerical calculation
            dx, dy = self.spline(s, 1)[:2]
            ddx, ddy = self.spline(s, 2)[:2]
            k = (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** (3 / 2))
            return k
        else:
            k = self.spline(s)[3]
            return k

    def calc_yaw(self, s: float) -> Optional[float]:
        """
        Calc yaw angle at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. If `s` is outside the data point's range, return None.

        Returns
        -------
        yaw : float
            yaw angle (tangent vector) for given s.
        """
        if self.spsi is None: # yaw was not provided => numerical calculation
            dx, dy = self.spline(s, 1)[:2]
            yaw = math.atan2(dy, dx)
            # Convert yaw to [0, 2pi]
            yaw = yaw % (2 * math.pi)
            return yaw
        else:
            yaw = self.spline(s)[2]
            return yaw
        
    def calc_arclength(self, x: float, y: float, s_guess=0.0) -> tuple[float, float]:
        """
        Fast calculation of arclength for a given point (x, y) on the trajectory.
        Less accuarate and less smooth than calc_arclength but much faster.
        Suitable for lap counting.

        Parameters
        ----------
        x : float
            x position.
        y : float
            y position.

        Returns
        -------
        s : float
            distance from the start point for given x, y.
        ey : float
            lateral deviation for given x, y.
        """
        _, ey, t, min_dist_segment = nearest_point_on_trajectory(
            np.array([x, y]).astype(np.float32), self.points[:, :2]
        )
        # s = s at closest_point + t
        s = float(
            self.s[min_dist_segment]
            + t * (self.s[min_dist_segment + 1] - self.s[min_dist_segment])
        )

        return s, ey

    def calc_arclength_slow(
        self, x: float, y: float, s_guess: float = 0.0
    ) -> tuple[float, float]:
        """
        Calculate arclength for a given point (x, y) on the trajectory.

        Parameters
        ----------
        x : float
            x position.
        y : float
            y position.
        s_guess : float
            initial guess for s.

        Returns
        -------
        s : float
            distance from the start point for given x, y.
        ey : float
            lateral deviation for given x, y.
        """

        def distance_to_spline(s):
            x_eval, y_eval = self.spline(s)[0]
            return np.sqrt((x - x_eval) ** 2 + (y - y_eval) ** 2)

        output = so.fmin(distance_to_spline, s_guess, full_output=True, disp=False)
        closest_s = float(output[0][0])
        absolute_distance = output[1]
        return closest_s, absolute_distance
    
    def _calc_tangent(self, s: float) -> np.ndarray:
        """
        Calculates the tangent to the curve at a given point.

        Parameters
        ----------
        s : float
            distance from the start point.
            If `s` is outside the data point's range, return None.

        Returns
        -------
        tangent : float
            tangent vector for given s.
        """
        dx, dy = self.spline(s, 1)[:2]
        tangent = np.array([dx, dy])
        return tangent

    def _calc_normal(self, s: float) -> np.ndarray:
        """
        Calculate the normal to the curve at a given point.

        Parameters
        ----------
        s : float
            distance from the start point.
            If `s` is outside the data point's range, return None.

        Returns
        -------
        normal : float
            normal vector for given s.
        """
        dx, dy = self.spline(s, 1)[:2]
        normal = np.array([-dy, dx])
        return normal

    def calc_velocity(self, s: float) -> Optional[float]:
        """
        Calc velocity at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        v : float
            velocity for given s.
        """
        if self.vxs is None: # velocity was not provided => numerical calculation
            dx, dy = self.spline(s, 1)[:2]
            v = np.hypot(dx, dy)
            return v 
        else:
            v = self.spline(s)[4]
            return v
        
    def calc_acceleration(self, s: float) -> Optional[float]:
        """
        Calc acceleration at the given s.

        Parameters
        ----------
        s : float
            distance from the start point. if `s` is outside the data point's
            range, return None.

        Returns
        -------
        a : float
            acceleration for given s.
        """
        if self.axs is None: # acceleration was not provided => numerical calculation
            ddx, ddy = self.spline(s, 2)[:2]
            a = np.hypot(ddx, ddy)
            return a
        else:
            a = self.spline(s)[5]
            return a
        
