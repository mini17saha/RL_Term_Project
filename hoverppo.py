import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import os
import time
import keyboard
import signal
import sys
from datetime import datetime
from stable_baselines3.common.policies import obs_as_tensor

from cflib import crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
import logging
from gym import spaces


class CrazyflieEnv(gym.Env):
    def __init__(self, URI='radio://0/80/2M/E7E7E7E7E7'):
        super(CrazyflieEnv, self).__init__()
        
        # Setup logging and initialize Crazyflie drivers
        # logging.basicConfig(level=logging.ERROR)
        crtp.init_drivers(enable_debug_driver=False)

        # Connect to Crazyflie
        self.URI = URI
        self.scf = SyncCrazyflie(self.URI, cf=Crazyflie(rw_cache="./cache"))
        self.scf.open_link()
        self._setup_logging()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1]),
                                       high=np.array([1, 1, 1, 1]),
                                       dtype=np.float32)
        # self.action_space = spaces.MultiDiscrete([10, 10, 10, 9])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # Environment state variables
        self.state = np.zeros(9)
        self.target_position = np.array([0, 0, 0.4], dtype = np.float64)
        self.max_steps = 1024
        self.current_step = 0
        # self.init_state = None

    def _setup_logging(self):
        """Set up logging for the Crazyflie to retrieve state data."""
        self.log_conf = LogConfig(name="Data", period_in_ms=50)
        self.log_conf.add_variable("stateEstimate.x", "float")
        self.log_conf.add_variable("stateEstimate.y", "float")
        self.log_conf.add_variable("stateEstimate.z", "float")
        self.log_conf.add_variable("stabilizer.roll", "float")
        self.log_conf.add_variable("stabilizer.pitch", "float")
        self.log_conf.add_variable("stabilizer.yaw", "float")
        # self.log_conf.add_variable("stateEstimate.vx", "float")
        # self.log_conf.add_variable("stateEstimate.vy", "float")
        # self.log_conf.add_variable("stateEstimate.vz", "float")

        
        self.scf.cf.log.add_config(self.log_conf)
        self.log_conf.data_received_cb.add_callback(self._log_callback)
        self.log_conf.start()

    def _log_callback(self, timestamp, data, log_conf):
        """Callback function to update state from Crazyflie logs."""
        # Update state with data from Crazyflie logs (example placeholders)
        # print('state updated')
        self.state[:3] = [data.get("stateEstimate.x", 0), data.get("stateEstimate.y", 0), data.get("stateEstimate.z", 0)]
        # print(data.get("stateEstimate.z", 0))
        # self.state[3:6] = [data.get("stateEstimate.vx", 0), data.get("stateEstimate.vy", 0), data.get("stateEstimate.vz", 0)]
        self.state[6:9] = [data.get("stabilizer.roll", 0), data.get("stabilizer.pitch", 0), data.get("stabilizer.yaw", 0)]
        # print(data)

    def _send_control_command(self, thrust, roll, pitch, yaw):
        """Convert action parameters and send control commands to Crazyflie."""
        self.scf.cf.commander.send_setpoint(0, 0, 0, 0)
        # print(f"{thrust},{roll},{pitch},{yaw}")
        
        roll = int(-45 + (roll * 10))
        pitch = int(-45 + (pitch * 10))
        self.scf.cf.commander.send_setpoint(roll, pitch, 0, thrust)
        time.sleep(0.01)

    def _get_state(self):
        """Retrieve the Crazyflie's current state (updated by the logging callback)."""
        return self.state

    def close(self):
        """Close the Crazyflie connection and stop logging."""
        self.log_conf.stop()
        self.scf.close_link()

class PhysicalCrazyflieEnvWrapper(CrazyflieEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_count = 0
        self.step_count = 0
        self.previous_states = []
        self.state_history_len = 5
        self.max_episode_steps = 1024  # Limit episode length for safety
        self.cum_reward = [0,0,0,0]
        self.total_reward = 0
        self.prev_thrust = 0
        

        signal.signal(signal.SIGINT, self._emergency_stop_handler)
        self.emergency_stop = False
        
    def _emergency_stop_handler(self, signum, frame):
        print("\nEmergency stop triggered! Landing drone...")
        self.emergency_stop = True
        self.force_stop()
        sys.exit(0)
        
    def force_stop(self):
        try:
            # Send zero thrust command
            self.scf.cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.1)
            self.scf.cf.commander.send_stop_setpoint()
        except Exception as e:
            print(f"Error during emergency stop: {e}")
        finally:
            self.close()

    def _calculate_reward(self):
        current_pos = self.state[:3]
        target_pos = self.target_position
        # print(current_pos - self.target_position)

        # print(current_pos[2])
        # distance = np.linalg.norm(current_pos - target_pos)
        position_reward = 0
        delx = abs(current_pos[0] - target_pos[0])
        dely = abs(current_pos[1] - target_pos[1])
        delz = abs(current_pos[2] - target_pos[2])
        position_reward -= (delz*20 + 5*delx + 5*dely)
        # if(delx>=0.1) :
        #     position_reward -= 5*delx
        # if(dely>=0.1) :
        #     position_reward -= 5*dely
        if delz > 0.2 :
            position_reward -= 20*delz
        if delx > 0.2:
            position_reward -= 30*delx
        if dely > 0.2:
            position_reward -= 30*dely
        if delx<0.1 and dely<0.1 and delz<0.1:
            position_reward += 35
        if delz == 0 :
            position_reward += 75

        self.cum_reward[0] -= delx
        self.cum_reward[1] -= dely
        self.cum_reward[2] -= delz
        
        velocities = self.state[3:6]
        velocity_penalty = -1.0 * np.linalg.norm(velocities)
        
        attitude = self.state[6:8]
        attitude_penalty = 0
        for angle in attitude:
            if angle > 60 or angle < -60: #make this 30 ig
                attitude_penalty = -10
        # attitude_penalty = -0.5 * (np.abs(attitude[0]) + np.abs(attitude[1]))

        self.cum_reward[3] += attitude_penalty
        
        stability_reward = 0
        if len(self.previous_states) >= 2:
            pos_changes = np.diff([s[:3] for s in self.previous_states], axis=0)
            jitter = np.mean(np.linalg.norm(pos_changes, axis=1))
            stability_reward = -jitter * 2.0
        
        self.previous_states.append(self.state.copy())
        if len(self.previous_states) > self.state_history_len:
            self.previous_states.pop(0)
        
        safety_bonus = 0  #initial state need to be saved ig
        if (abs(delx) > 0.75 and 
            abs(dely) >0.75 and 
            abs(delz) >1.25):
            safety_bonus = -20

        # if episode_count < 10 and thrust < 0.5:
        #     position_reward += -10
        
        total_reward = (
            position_reward +
            # velocity_penalty +
            attitude_penalty +
            # stability_reward 
            safety_bonus
        )
        
        return total_reward

    def reset(self):
        """Reset the Crazyflie environment to the initial state."""
        print("Total Ep reward:",self.cum_reward)
        print("Total reward avg: ",self.total_reward/ (self.step_count + 1))
        self.scf.cf.commander.send_stop_setpoint()
        self.cum_reward =[0,0,0,0]
        # Close the link to start fresh
        self.scf.close_link()
        
        # Notify user and wait for them to set up the drone
        print(f"\nEpisode {self.episode_count} completed.")
        print("Please place the drone in the starting position and ensure the area is clear.")
        input("Press Enter when ready to start the next episode...")
        
        # Re-open link and set up logging again
        self.scf.open_link()
        self._setup_logging()  # Re-establish logging after re-opening the link
        time.sleep(2)
        self.init_state = self.state.copy()
        print(self.init_state)
        self.target_position[:2] = self.init_state[:2]
        self.target_position[2] = 0.4
        # Send stop command to ensure drone is in idle state
        time.sleep(2.0)  # Wait for physical setup

        # Reset episode variables
        self.previous_states = []
        self.step_count = 0
        self.episode_count += 1
        
        # Initialize the state and wait for the logging callback to update it
        self.state = np.zeros(9)
        time.sleep(0.1)  # Give time for initial state estimation
        
        # Send a small command to stabilize state estimation
        self._send_control_command( 0, 0, 0, 0)
        time.sleep(0.1)
    
        return self.state

    def step(self, action):
        # print(self.init_state - self.state)
        # print(self.state)
        if self.emergency_stop:
            return self.state, 0, True, {"emergency_stop": True}
        
        # Increment step counter
        self.step_count += 1
        
        # Clip actions for safety
        # print(action)
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        
        
        # Send commands to drone
        thrust, roll, pitch, yaw = action
        n=1
        # if(self.step_count<=20) :
        #     thrust_value = int(20000 + (thrust* 5000))
        # else :
        #     thrust_value = int(10000 + (thrust* 5000))
        # thrust_value = int(20000*np.exp(-0.001*self.step_count) + (thrust* 5000))
        thrust_value = int(15000*np.exp(-0.001*self.step_count)+(thrust* 5000))
        
        thrust_value = min(65000, thrust_value)
        # if self.step_count<=15 : 
        #     thrust = 55000
        #     self.prev_thrust = thrust
        # else :
        #     thrust=self.prev_thrust - 35
        #     self.prev_thrust = thrust
        self._send_control_command( thrust_value, roll/n, pitch/n, yaw/n)
        
        # Update state
        self.state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Check termination conditions
        done = self._is_done()
        if done[0]: reward -= 10000
        done = done[0] or done[1]
        
        # Additional info
        info = {
            "step": self.step_count,
            "episode_len": self.episode_count,
            "state": self.state,
            "action": action
        }
        # print(reward)
        # self.cum_reward += reward
        return self.state, reward, done, info

    def _is_done(self):
        pos = self.state[:3]
        diff = abs(self.state - self.init_state)
        out_of_bounds = (
            diff[0] > 0.75 or
            diff[1] > 0.75 or
            diff[2] > 1.25
        )
        
        step_limit_reached = self.step_count >= self.max_episode_steps
        if out_of_bounds:
            print("Out of bounds!")
            print(diff[0],diff[1],diff[2])
        if step_limit_reached:
            print("Step limit reached!")
        return [out_of_bounds ,step_limit_reached]

def train_ppo_physical(total_timesteps=50000, save_dir='./cache/ppo_physical_crazyflie'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment
    env = PhysicalCrazyflieEnvWrapper()
    env = DummyVecEnv([lambda: env])
    
    # Initialize PPO with conservative hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.005,      # Lower learning rate for stability
        n_steps=256,            # Smaller batch size
        batch_size=16,
        n_epochs=5,              # Fewer epochs per update
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.1,          # Conservative clip range
        ent_coef=0.01,
        tensorboard_log=save_dir
    )

    # print("TRAINED")
    save_dir='./cache/ppo_physical_crazyflie'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"{save_dir}"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=16, save_replay_buffer=True, save_path=save_dir)
    try:
        print("\nStarting training. Press Ctrl+C for emergency stop.")
        print("Ensure the physical space is clear and the drone is in the starting position.")
        input("Press Enter to begin...")
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback= checkpoint_callback,
            tb_log_name="ppo_physical"
        )
        
        # Save final model
        model.save(f"{save_dir}/final_model")
        loaded_model = PPO.load(f"{save_dir}/final_model")
        loaded_model.save_replay_buffer("hover_replay_buffer")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving model...")
        model.save(f"{save_dir}/interrupted_model")
        env.get_attr('force_stop')[0]()
    
    return model, env

def resume_training(model_path, env, total_timesteps=50000, load_replay=False):
    """
    Resume training from a saved model
    """
    model = PPO.load(model_path, env=env)
    # if load_replay:
    #     model.load_replay_buffer("hover_replay_buffer")
    # return train_ppo_physical(total_timesteps=timesteps, 
    #                         save_dir=os.path.dirname(model_path))

    save_dir='./cache/ppo_physical_crazyflie'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"{save_dir}"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=16, save_replay_buffer=True, save_path=save_dir)
    try:
        print("\nStarting training. Press Ctrl+C for emergency stop.")
        print("Ensure the physical space is clear and the drone is in the starting position.")
        input("Press Enter to begin...")
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback= checkpoint_callback,
            tb_log_name="ppo_physical"
        )
        
        # Save final model
        model.save(f"{save_dir}/final_model")
        model.save_replay_buffer("hover_replay_buffer")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving model...")
        model.save(f"{save_dir}/interrupted_model")
        env.get_attr('force_stop')[0]()
    
    return model, env

if __name__ == "__main__":
    # Check for saved model to resume training
    latest_model = None
    if len(sys.argv) > 1:
        latest_model = sys.argv[1]
        print(f"Resuming training from {latest_model}")
        env = PhysicalCrazyflieEnvWrapper()
        env = DummyVecEnv([lambda: env])
        model, env = resume_training(latest_model, env, load_replay=True)

    else:
        print("Starting new training session")
        model, env = train_ppo_physical()