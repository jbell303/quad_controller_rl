"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Combined(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]), dtype=np.float32)
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 5.0  # secs
        self.weight_position = 0.3
        self.weight_velocity = 0.2
        self.task_stage = "takeoff"
        print("Starting takeoff...")

    def reset(self):
        # Reset episode-specific variables
        self.last_timestamp = None
        self.last_position = None
        self.task_stage = "takeoff"
        print("starting takeoff...")
        
        # Return initial condition
        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, 
            pose.orientation.z, pose.orientation.w])
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03) # prevent divide by zero
        state = np.concatenate([position, orientation, velocity]) # combined state vector
        self.last_timestamp = timestamp
        self.last_position = position
        # print("angular_velocity: {}, linear_acceleration: {}".format(angular_velocity, linear_acceleration))

        done = False

        if self.task_stage == "takeoff":
            # Task-specific parameters
            self.target_z = 10.0  # target height (z position) to reach for successful takeoff
            self.target_velocity_z = 5.0
            
            # Compute task-specific error
            error_position = abs(self.target_z - pose.position.z)
            error_velocity = (self.target_velocity_z - state[9])**2
            
            # Compute reward / penalty and check if this episode is complete
            reward = -min(self.weight_position * error_position + self.weight_velocity * error_velocity, 20.0) # reward = zero for matching target z, -ve as you go farther, upto -20
            if pose.position.z >= self.target_z:  # agent has crossed the target height
                reward += 10.0  # bonus reward
                self.task_stage = "hover"
                print("starting hover...")
            elif timestamp > self.max_duration:  # agent has run out of time
                reward -= 10.0  # extra penalty
                done = True
                print("ending takeoff (timeout)...")

        if self.task_stage == "hover":
            # Task-specific parameters
            self.max_duration += 5.0  # secs
            self.max_error_position = 8.0 # distance units
            self.target_position = np.array([0.0, 0.0, 10.0]) # target position to hover at
            self.weight_position = 0.5
            self.target_velocity = np.array([0.0, 0.0, 0.0]) # target velocity (ideally stary in place)
            self.weight_velocity = 0.3
            
            # Compute task-specific error
            error_position = np.linalg.norm(self.target_position - state[0:3]) # Euclidian distance from target position vector
            error_velocity = np.linalg.norm(self.target_velocity - state[7:10])**2
            
            # Compute reward / penalty and check if this episode is complete
            reward = -(self.weight_position * error_position + self.weight_velocity * error_velocity)
            if error_position > self.max_error_position:
                reward -= 50.0  # extra penalty, agent strayed too far
                done = True
                print("ending hover(max error position: {})...".format(state[2]))
            elif timestamp > self.max_duration:  # agent has run out of time
                reward += 50.0  # extra reward, agent made it to the end
                self.task_stage = "landing"
                print("starting landing...")

        if self.task_stage == "landing":
            # Task-specific parameters
            self.max_duration += 10.0  # secs
            self.max_error_position = 15.0 # distance units
            self.target_position = np.array([0.0, 0.0, 0.1]) # target position to land
            self.weight_position = 0.5
            self.target_velocity = np.array([0.0, 0.0, -2.0]) # target velocity (we want a soft landing)
            self.weight_velocity = 0.3

            error_position = np.log(np.linalg.norm(self.target_position - state[0:3])) # Euclidian distance from target position vector
            error_velocity = np.linalg.norm(self.target_velocity - state[7:10])**2
            # Compute reward / penalty and check if this episode is complete
            reward = -(self.weight_position * error_position + self.weight_velocity * error_velocity)
            if error_position > self.max_error_position:
                reward -= 50.0  # extra penalty, agent strayed too far
                done = True
                print("ending landing (max error position: {})...".format(state[2]))
            elif timestamp > self.max_duration:  # agent has run out of time
                reward -= 10.0  # extra penalty, agent was too slow
                done = True
                print("ending landing (timeout)...")
            elif pose.position.z <= self.target_position[2]: # agent landed
                reward += 50.0 # extra reward, agent landed
                done = True
                print("landing complete. Task accomplished!")

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
