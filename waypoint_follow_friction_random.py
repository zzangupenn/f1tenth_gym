import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import json

from numba import njit

from pyglet.gl import GL_POINTS

import pdb

"""
Planner Helpers
"""


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
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
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)
    steering_angle = np.arctan(wheelbase / radius)

    # speed = vel_

    return speed, steering_angle


class PurePursuitPlanner:
    """
    Example Planner
    """

    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
        # self.waypoints = self.waypoints[::-1] # NOTE: reverse map
        # self.waypoints = self.waypoints * .1 # NOTE: map scales

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        # points = self.waypoints

        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T

        scaled_points = 50. * points
        # scaled_points = 1. * points # TODO

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts,
                                                                                    i + t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3,))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle
    

def get_steers():
    scale_ = 10 # 1 for 0.001 time step
    length = int(1e6 // scale_)
    peaks = 200

    x = np.linspace(0, 1, length)
    y = np.zeros_like(x)

    for _ in range(peaks):
        amplitude = np.random.rand() 
        frequency = np.random.randint(1, peaks)
        phase = np.random.rand() * 2 * np.pi 

        y += amplitude * np.sin(2 * np.pi * frequency * x + phase)

    y -= np.mean(y)
    y_lower = np.min(y)
    z = y - y_lower
    y_upper = np.max(z)
    z = z/y_upper
    z = z*2
    z = z - 1.

    return z


def warm_up(conf, steers_, num_of_sim_steps):
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext,
                        num_agents=1, timestep=0.01, model='MB', drive_control_mode='vel',
                        steering_control_mode='angle')

    # init vector = [x,y,yaw,steering angle, velocity, yaw_rate, beta]
    obs, step_reward, done, info = env.reset(
        np.array([[conf.sx, conf.sy, conf.stheta, 0.0, 0.0, 0.0, 0.0]]))

    step_count_ = 0
    warm_up_steps = 2000

    while step_count_ < warm_up_steps:
        steer = steers_[step_count_]

        env.params['tire_p_dy1'] = friction_  # mu_y
        env.params['tire_p_dx1'] = friction_  # mu_x

        step_reward = 0.0

        for i in range(num_of_sim_steps):
            step_count_ += 1

            try:
                obs, rew, done, info = env.step(np.array([[steer, vel_]]))
            except ZeroDivisionError:
                print('error at: ', step_count_)
                return 'error', ' ', ' ', ' '

    return step_count_, env, obs, step_reward




vel_ = 8.
friction_ = 1.0

def main():
    """
    main entry point
    """
    
    with open('maps/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)


    vels_ = [7.7]
    frictions_ = [0.6, 1.0, 1.4]
    # frictions_ = [1.4]
    steers_ = get_steers()
    for vel_ in vels_:
        for friction_ in frictions_:
            
            
            start = time.time()

            num_of_sim_steps = 20 # NOTE: fixed no matter what env.timestep is
            
            total_states = None
            total_controls = None
            for _ in range(1): # more samples
                states = None
                controls = None
                # steers_ = get_steers()

                step_count_ = 'error'
                while step_count_ == 'error':
                    step_count_, env, obs, step_reward = warm_up(conf, steers_, num_of_sim_steps)

                while step_count_ < len(steers_):
                    print(step_count_, '/', len(steers_))
                    steer = steers_[step_count_]

                    env.params['tire_p_dy1'] = friction_  # mu_y
                    env.params['tire_p_dx1'] = friction_  # mu_x

                    step_reward = 0.0

                    for i in range(num_of_sim_steps):
                        step_count_ += 1

                        try:
                            obs, rew, done, info = env.step(np.array([[steer, vel_]]))
                        except ZeroDivisionError:
                            print('error at: ', step_count_)
                            step_count_ += num_of_sim_steps - i - 1
                            steers_ = get_steers()
                            check_error = 'error'
                            while check_error == 'error':
                                check_error, env, obs, _ = warm_up(conf, steers_, num_of_sim_steps)
                            break

                        state = np.array([obs['x3'][0], obs['x4'][0], obs['x6'][0], 
                                        obs['x11'][0]])
                        #x3 = steering angle of front wheels
                        #x4 = velocity in x-direction
                        #x6 = yaw rate
                        #x11 = velocity in y-direction

                        control = np.array([steer, vel_])

                        if states is None:
                            states = state.reshape((1, state.shape[0]))
                        else:
                            states = np.vstack((states, state.reshape((1, state.shape[0]))))
                        if controls is None:
                            controls = control.reshape((1, control.shape[0]))
                        else:
                            controls = np.vstack((controls, control.reshape((1, control.shape[0]))))

                    # # NOTE: render
                    # env.render(
                    #     mode='human_fast')  # Naive implementation of 'human' render mode does not work well, use 'human_fast'

                if total_states is None:
                    total_states = states
                else:
                    total_states = np.vstack((total_states, states))
                if total_controls is None:
                    total_controls = controls
                else:
                    total_controls = np.vstack((total_controls, controls))

            file_name_ = 'data/Random_Spielberg_raceline/' # NOTE
            np.save(file_name_+'states_mb_fric_{}_vel_{}.npy'.format(int(friction_*10), int(vel_*10)), total_states)
            np.save(file_name_+'controls_mb_fric_{}_vel_{}.npy'.format(int(friction_*10), int(vel_*10)), total_controls)

            print('Real elapsed time:', time.time() - start)


if __name__ == '__main__':
    main()
