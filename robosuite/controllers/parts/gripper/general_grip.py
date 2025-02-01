"""This is a controller that controls the fingers / grippers to do naive gripping. No matter how many fingers the gripper has, they all move in the same direction."""

import numpy as np
import traceback

from robosuite.controllers.parts.gripper.gripper_controller import GripperController
from robosuite.utils.control_utils import *

# Supported impedance modes

# Supports position / velocity / torque control with target position as goal


class GeneralGripController(GripperController):
    """
    Controller for controlling gripper via impedance control.

    NOTE: Control input actions assumed to be the target joint positions. A given action to this
    controller is assumed to be of the form: (pos_j0, pos_j1, ... , pos_jn-1) for an n-joint robot

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or Iterable of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or Iterable of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or Iterable of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or Iterable of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        qpos_limits (2-list of float or 2-list of Iterable of floats): Limits (rad) below and above which the magnitude
            of a calculated goal joint position will be clipped. Can be either be a 2-list (same min/max value for all
            joint dims), or a 2-list of list (specific min/max values for each dim)

        interpolator (Interpolator): Interpolator object to be used for interpolating from the current joint position to
            the goal joint position during each timestep between inputted actions

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error

    Raises:
        AssertionError: [Invalid impedance mode]
    """

    def __init__(
        self,
        sim,
        joint_indexes,
        actuator_range,
        input_max=1,
        input_min=-1,
        output_max=1,
        output_min=-1,
        policy_freq=20,
        qpos_limits=None,
        interpolator=None,
        use_action_scaling=True,
        control_mode="torque",
        kp=200,
        ki=0.1,
        kd=10,
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):
        super().__init__(
            sim,
            joint_indexes,
            actuator_range,
            part_name=kwargs.get("part_name", None),
            naming_prefix=kwargs.get("naming_prefix", None),
        )

        # Control dimension
        self.control_dim = len(joint_indexes["actuators"])

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # limits
        self.position_limits = np.array(qpos_limits) if qpos_limits is not None else qpos_limits

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # action scaling
        self.use_action_scaling = use_action_scaling

        # initialize
        self._goal_qpos = None
        self.qpos = None
        self.vels = None
        self.torques = None

        self.control_mode = control_mode

        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Initialize integral error
        self.integral_error = np.zeros(self.control_dim)
        self.dt = 1.0 / policy_freq

    @property
    def goal_qpos(self):
        return self._goal_qpos
        
    @goal_qpos.setter
    def goal_qpos(self, value):
        # print("\n[DEBUG] Gripper goal_qpos being set to:", value)
        # print("Stack trace:")
        # traceback.print_stack()
        self._goal_qpos = value

    def set_goal(self, action, set_qpos=None):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
            set_qpos (Iterable): If set, overrides @action and sets the desired absolute joint position goal state

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        
        if action is None:  # no action, do nothing
            return
        # Update state
        self.update()

        # Check to make sure delta is size self.joint_dim
        assert len(action) == self.control_dim, (
            f"Delta qpos must be equal to the control dimension of the robot!"
            f"Expected {self.control_dim}, got {len(action)}"
        )

        # scaled_delta = action
        # if self.use_action_scaling:
        #     scaled_delta = self.scale_action(action)

        self.goal_qpos = action

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qpos)

        # if 11 in self.qpos_index:
        #     print(action)
        #     print(self.goal_qpos)

    def run_controller(self):
        """
        Calculates the control command required to reach the desired setpoint

        Returns:
             np.array: Command control command, can be position, velocity, or torque
        """
        # Make sure goal has been set
        if self.goal_qpos is None:
            self.set_goal(np.zeros(self.control_dim))

        # Update state
        self.update()

        desired_qpos = None

        # Only linear interpolator is currently supported
        if self.interpolator is not None:
            # Linear case
            if self.interpolator.order == 1:
                desired_qpos = self.interpolator.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            desired_qpos = np.array(self.goal_qpos)

        # Calculate errors
        position_error = desired_qpos - self.joint_pos
        velocity_error = -self.joint_vel

        # Update integral error
        self.integral_error += position_error * self.dt

        if self.control_mode == "torque":
            # PID control
            p_term = np.multiply(position_error, self.kp)
            i_term = np.multiply(self.integral_error, self.ki)
            d_term = np.multiply(velocity_error, self.kd)
            
            desired_torque = p_term + i_term + d_term
            self.torques = np.dot(self.mass_matrix, desired_torque) + self.torque_compensation
        elif self.control_mode == "velocity":
            # TODO: under testing
            desired_qvel = self.kp * position_error
            self.vels = desired_qvel
            if self.use_action_scaling:
                ctrl_range = np.stack([self.actuator_min, self.actuator_max], axis=-1)
                bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
                weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
                self.vels = bias + weight * desired_qvel
        elif self.control_mode == "position":
            # TODO: under testing
            desired_qpos = np.array(self.goal_qpos)
            self.qpos = desired_qpos
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")

        # Always run superclass call for any cleanups at the end
        super().run_controller()
        if self.control_mode == "velocity":
            return self.vels
        elif self.control_mode == "torque":
            return self.torques
        elif self.control_mode == "position":
            return self.qpos

    def reset_goal(self):
        """
        Resets joint position goal to be current position
        """
        self.goal_qpos = self.joint_pos

        # Reset integral error
        self.integral_error = np.zeros(self.control_dim)

        # Reset interpolator if required
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qpos)

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property
        Returns the following (generalized for both high and low limits), based on the impedance mode:
        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        return self.input_min, self.input_max

    @property
    def name(self):
        return "JOINT_POSITION"
