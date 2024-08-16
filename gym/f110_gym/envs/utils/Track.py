from __future__ import annotations
import pathlib
from typing import Optional
import numpy as np
from .cubic_spline import CubicSplineND


class Track:
    ss: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    vxs: np.ndarray
    centerline: CubicSplineND
    raceline: CubicSplineND
    filepath: Optional[str]
    ss: Optional[np.ndarray] = None
    psis: Optional[np.ndarray] = None
    kappas: Optional[np.ndarray] = None
    accxs: Optional[np.ndarray] = None

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        velxs: np.ndarray,
        filepath: Optional[str] = None,
        centerline: Optional[CubicSplineND] = None,
        raceline: Optional[CubicSplineND] = None,
        ss: Optional[np.ndarray] = None,
        psis: Optional[np.ndarray] = None,
        kappas: Optional[np.ndarray] = None,
        accxs: Optional[np.ndarray] = None,
    ):
        """
        Initialize track object.

        Parameters
        ----------
        spec : TrackSpec
            track specification
        filepath : str
            path to the track image
        occupancy_map : np.ndarray
            occupancy grid map
        centerline : Raceline, optional
            centerline of the track, by default None
        raceline : Raceline, optional
            raceline of the track, by default None
        """
        self.filepath = filepath

        assert xs.shape == ys.shape == velxs.shape, "inconsistent shapes for x, y, vel"

        self.n = xs.shape[0]
        self.ss = ss
        self.xs = xs
        self.ys = ys
        self.yaws = psis
        self.ks = kappas
        self.vxs = velxs
        self.axs = accxs

        # approximate track length by linear-interpolation of x,y waypoints
        # note: we could use 'ss' but sometimes it is normalized to [0,1], so we recompute it here
        self.length = float(np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)))

        self.centerline = centerline or CubicSplineND(xs, ys, psis, kappas, velxs, accxs)
        self.raceline = raceline or CubicSplineND(xs, ys, psis, kappas, velxs, accxs)
        self.s_guess = 0.0

    @staticmethod
    def from_numpy(waypoints: np.ndarray, downsample_step = 1) -> Track:
        """
        Create an empty track reference line.

        Parameters
        ----------
        filepath : pathlib.Path
            path to the raceline file
        delimiter : str, optional
            delimiter used in the file, by default ";"
        downsample_step : int, optional
            downsample step for waypoints, by default 1 (no downsampling)

        Returns
        -------
        track: Track
            track object
        """
        assert (
            waypoints.shape[1] == 7
        ), "expected waypoints as [s, x, y, psi, k, vx, ax]"
        
        ss=waypoints[::downsample_step, 0]
        xs=waypoints[::downsample_step, 1]
        ys=waypoints[::downsample_step, 2]
        yaws=waypoints[::downsample_step, 3]
        ks=waypoints[::downsample_step, 4]
        vxs=waypoints[::downsample_step, 5]
        axs=waypoints[::downsample_step, 6]

        refline = CubicSplineND(xs, ys, yaws, ks, vxs, axs)

        return Track(
            xs=xs,
            ys=ys,
            velxs=vxs,
            ss=refline.s,
            psis=yaws,
            kappas=ks,
            accxs=axs,
            filepath=None,
            raceline=refline,
            centerline=refline,
        )
    
    @staticmethod
    def from_raceline_file(filepath: pathlib.Path, delimiter: str = ";", downsample_step = 1) -> Track:
        """
        Create an empty track reference line.

        Parameters
        ----------
        filepath : pathlib.Path
            path to the raceline file
        delimiter : str, optional
            delimiter used in the file, by default ";"
        downsample_step : int, optional
            downsample step for waypoints, by default 1 (no downsampling)

        Returns
        -------
        track: Track
            track object
        """
        assert filepath.exists(), f"input filepath does not exist ({filepath})"
        waypoints = np.loadtxt(filepath, delimiter=delimiter).astype(np.float32)
        assert (
            waypoints.shape[1] == 7
        ), "expected waypoints as [s, x, y, psi, k, vx, ax]"
        
        ss=waypoints[::downsample_step, 0]
        xs=waypoints[::downsample_step, 1]
        ys=waypoints[::downsample_step, 2]
        yaws=waypoints[::downsample_step, 3]
        ks=waypoints[::downsample_step, 4]
        vxs=waypoints[::downsample_step, 5]
        axs=waypoints[::downsample_step, 6]

        refline = CubicSplineND(xs, ys, yaws, ks, vxs, axs)

        return Track(
            xs=xs,
            ys=ys,
            velxs=vxs,
            ss=refline.s,
            psis=yaws,
            kappas=ks,
            accxs=axs,
            filepath=filepath,
            raceline=refline,
            centerline=refline,
        )
    
    def frenet_to_cartesian(self, s, ey, ephi):
        """
        Convert Frenet coordinates to Cartesian coordinates.

        s: distance along the raceline
        ey: lateral deviation
        ephi: heading deviation

        returns:
            x: x-coordinate
            y: y-coordinate
            psi: yaw angle
        """
        x, y = self.centerline.calc_position(s)
        psi = self.centerline.calc_yaw(s)

        # Adjust x,y by shifting along the normal vector
        x -= ey * np.sin(psi)
        y += ey * np.cos(psi)

        # Adjust psi by adding the heading deviation
        psi += ephi

        return x, y, psi

    def cartesian_to_frenet(self, x, y, phi, s_guess=None):
        """
        Convert Cartesian coordinates to Frenet coordinates.

        x: x-coordinate
        y: y-coordinate
        phi: yaw angle

        returns:
            s: distance along the centerline
            ey: lateral deviation
            ephi: heading deviation
        """
        if s_guess is None: # Utilize internal state to keep track of the guess
            s_guess = self.s_guess

        s, ey = self.centerline.calc_arclength(x, y, s_guess)
        # Wrap around
        s = s % self.centerline.s[-1]

        self.s_guess = s # Update the guess for the next iteration

        # Use the normal to calculate the signed lateral deviation
        normal = self.centerline._calc_normal(s)
        x_eval, y_eval = self.centerline.calc_position(s)
        dx = x - x_eval
        dy = y - y_eval
        distance_sign = np.sign(np.dot([dx, dy], normal))
        ey = ey * distance_sign

        phi = phi - self.centerline.calc_yaw(s)

        return s, ey, phi

    def curvature(self, s):
        """
        Get the curvature at a given s.

        s: distance along the raceline

        returns:
            curvature
        """
        return self.centerline.calc_curvature(s)
    
if __name__ == "__main__":
    # Load the racline.csv
    track = Track.from_raceline_file(
        filepath=pathlib.Path("raceline.csv"), delimiter=";", downsample_step=10
    )
    waypoints = np.hstack((track.raceline.s.reshape(-1,1), track.raceline.points))
    track = Track.from_numpy(waypoints, downsample_step=1)

    # Get a random s, ey point and plot it colored by curvature
    POINTS_TO_TEST = 1000
    s = np.random.uniform(0, track.raceline.s[-1], size=(POINTS_TO_TEST))
    ey = np.random.uniform(-2, 2, size=(POINTS_TO_TEST))
    ephi = 0
    points = np.array([s, ey]).T
    X, Y, CURV = [], [], []
    VXS, AXS = [], []
    for point in points:
        s_, ey_ = point
        x, y, psi = track.frenet_to_cartesian(s_, ey_, ephi)
        curv = track.curvature(s_)
        # print(f"s: {s}, ey: {ey}, ephi: {ephi}, x: {x}, y: {y}, psi: {psi}, curv: {curv}")
        X.append(x)
        Y.append(y)
        CURV.append(curv)
        vxs = track.raceline.calc_velocity(s_)
        axs = track.raceline.calc_acceleration(s_)
        VXS.append(vxs)
        AXS.append(axs)

    x, y = np.array(X), np.array(Y)
    CURV = np.array(CURV)
    VXS, AXS = np.array(VXS), np.array(AXS)

    # Scatter raceline xs, ys colored by curvature
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(track.raceline.xs, track.raceline.ys, c=track.raceline.ks)
    bar = plt.colorbar(label="Curvature", extend='both')
    plt.scatter(x, y, c=CURV, edgecolors='black', vmin=bar.get_ticks().min(), vmax=bar.get_ticks().max())
    plt.title("Curvature Sampled vs Raceline")
    
    plt.figure()
    plt.scatter(track.raceline.xs, track.raceline.ys, c=track.raceline.vxs)
    bar = plt.colorbar(label="Velocity", extend='both')
    plt.scatter(x, y, c=VXS, edgecolors='black', vmin=bar.get_ticks().min(), vmax=bar.get_ticks().max())
    plt.title("Velocity Sampled vs Raceline")

    fig = plt.figure()
    plt.scatter(track.raceline.xs, track.raceline.ys, c=track.raceline.axs)
    bar = plt.colorbar(label="Acceleration", extend='both')
    plt.scatter(x, y, c=AXS, edgecolors='black', vmin=bar.get_ticks().min(), vmax=bar.get_ticks().max())
    plt.title("Acceleration Sampled vs Raceline")
    plt.show()
    
    # Test out frenet to cartesian and cartesian to frenet
    s = 0
    ey = 0
    ephi = 0
    x, y, psi = track.frenet_to_cartesian(s, ey, ephi)
    s_, ey_, ephi_ = track.cartesian_to_frenet(x, y, psi)
    assert np.isclose(s, s_), f"Expected s: {s}, got: {s_}"
    assert np.isclose(ey, ey_), f"Expected ey: {ey}, got: {ey_}"
    assert np.isclose(ephi, ephi_), f"Expected ephi: {ephi}, got: {ephi_}"
    print("Simple Frenet to Cartesian and Cartesian to Frenet tests passed!")

    # Test out frenet to cartesian and cartesian to frenet
    x = 0
    y = 0
    psi = 0
    s, ey, ephi = track.cartesian_to_frenet(x, y, psi)
    x_, y_, psi_ = track.frenet_to_cartesian(s, ey, ephi)
    assert np.isclose(x, x_), f"Expected x: {x}, got: {x_}"
    assert np.isclose(y, y_), f"Expected y: {y}, got: {y_}"
    assert np.isclose(psi, psi_), f"Expected psi: {psi}, got: {psi_}"
    print("Simple Cartesian to Frenet and Frenet to Cartesian tests passed!")
