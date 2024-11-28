import gym
import logging
import time
import numpy as np
from gym import spaces
import cflib
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

class CrazyflieEnv(gym.Env):
    def __init__(self, URI='radio://0/80/2M/E7E7E7E7E7'):
        super(CrazyflieEnv, self).__init__()
        
        # Setup logging and initialize Crazyflie drivers
        logging.basicConfig(level=logging.ERROR)
        cflib.crtp.init_drivers(enable_debug_driver=False)

        # Connect to Crazyflie
        self.URI = URI
        self.scf = SyncCrazyflie(self.URI, cf=Crazyflie(rw_cache="./cache"))
        self.scf.open_link()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([0, -1, -1, -1]),
                                       high=np.array([1, 1, 1, 1]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # Environment state variables
        self.state = np.zeros(9)
        self.target_position = np.array([0, 0, 1])
        self.max_steps = 1000
        self.current_step = 0

    def _setup_logging(self):
        """Set up logging for the Crazyflie to retrieve state data."""
        self.log_conf = LogConfig(name="Data", period_in_ms=100)
        # Example: self.log_conf.add_variable("stateEstimate.x", "float")
        
        self.scf.cf.log.add_config(self.log_conf)
        self.log_conf.data_received_cb.add_callback(self._log_callback)
        self.log_conf.start()

    def _log_callback(self, timestamp, data, log_conf):
        """Callback function to update state from Crazyflie logs."""
        # Update state with data from Crazyflie logs (example placeholders)
        self.state[:3] = [data.get("stateEstimate.x", 0), data.get("stateEstimate.y", 0), data.get("stateEstimate.z", 0)]

    def reset(self):
        """Reset the Crazyflie and environment to the initial state."""
        self.scf.cf.commander.send_setpoint(0, 0, 0, 0)
        time.sleep(1)
        self.state = np.zeros(9)
        self.current_step = 0
        return self.state

    def _send_control_command(self, thrust, roll, pitch, yaw):
        """Convert action parameters and send control commands to Crazyflie."""
        self.scf.cf.commander.send_setpoint(0, 0, 0, 0)
        thrust_value = int(25000 * thrust + 20000)
        self.scf.cf.commander.send_setpoint(roll, pitch, yaw, thrust_value)
        time.sleep(0.1)

    def step(self, action):
        """Take an action, send it to the Crazyflie, and update the state."""
        thrust, roll, pitch, yaw = action
        self._send_control_command(thrust, roll, pitch, yaw)
        self.state = self._get_state()
        reward = self._calculate_reward()
        done = self._is_done()
        self.current_step += 1
        info = {"step": self.current_step, "distance_to_target": np.linalg.norm(self.state[:3] - self.target_position)}
        return self.state, reward, done, info

    def _get_state(self):
        """Retrieve the Crazyflie's current state (updated by the logging callback)."""
        return self.state

    def _calculate_reward(self):
        """Define the reward function."""
        distance_to_target = np.linalg.norm(self.state[:3] - self.target_position)
        return -distance_to_target

    def _is_done(self):
        """Define the episode termination conditions."""
        return self.current_step >= self.max_steps or not (-5 <= self.state[0] <= 5 and -5 <= self.state[1] <= 5 and 0 <= self.state[2] <= 10)

    def close(self):
        """Close the Crazyflie connection and stop logging."""
        self.log_conf.stop()
        self.scf.close_link()

# Testing the environment
if __name__ == "__main__":
    env = CrazyflieEnv()
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Replace with policy-generated actions
        state, reward, done, info = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
    env.close()
