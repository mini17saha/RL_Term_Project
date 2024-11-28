""" 
    Group Number: 3
    Roll Numbers: Yash Sirvi 21CS10083
                  Adyan Rizvi 21MA10006
                  Sreejita Saha 21CS30052
                  Soumojit Bhattacharya 21EC10071
                  Allen Emmanuel Binny 21EC30067
                  Anurag Sharma 24AI91R01
                  
    Project Number: 3
    Project Title:  Learning Safe Flight Manoeuvres for a Mini-Drone
    
"""



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
from stable_baselines3.common.buffers import ReplayBuffer
import pickle


class CrazyflieEnv(gym.Env):
    def __init__(self, URI='radio://0/80/2M/E7E7E7E7E7'):
        super(CrazyflieEnv, self).__init__()
        
        # Setup logging and initialize Crazyflie drivers
        crtp.init_drivers(enable_debug_driver=False)

        # Connect to Crazyflie
        self.URI = URI
        self.scf = SyncCrazyflie(self.URI, cf=Crazyflie(rw_cache="./cache"))
        self.scf.open_link()
        self._setup_logging()
        
        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([10, 10, 10, 9])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # Environment state variables
        self.state = np.zeros(9)
        self.target_position = np.array([0, 0, 0.4], dtype = np.float64)
        self.max_steps = 1024
        self.current_step = 0

    def _setup_logging(self):
        """Set up logging for the Crazyflie to retrieve state data."""
        self.log_conf = LogConfig(name="Data", period_in_ms=50)
        self.log_conf.add_variable("stateEstimate.x", "float")
        self.log_conf.add_variable("stateEstimate.y", "float")
        self.log_conf.add_variable("stateEstimate.z", "float")
        self.log_conf.add_variable("stabilizer.roll", "float")
        self.log_conf.add_variable("stabilizer.pitch", "float")
        self.log_conf.add_variable("stabilizer.yaw", "float")

        
        self.scf.cf.log.add_config(self.log_conf)
        self.log_conf.data_received_cb.add_callback(self._log_callback)
        self.log_conf.start()

    def _log_callback(self, timestamp, data, log_conf):
        """Callback function to update state from Crazyflie logs."""
        # Update state with data from Crazyflie logs (example placeholders)
        self.state[:3] = [data.get("stateEstimate.x", 0), data.get("stateEstimate.y", 0), data.get("stateEstimate.z", 0)]
        self.state[6:9] = [data.get("stabilizer.roll", 0), data.get("stabilizer.pitch", 0), data.get("stabilizer.yaw", 0)]

    def _send_control_command(self, thrust, roll, pitch, yaw):
        """Convert action parameters and send control commands to Crazyflie."""
        self.scf.cf.commander.send_setpoint(0, 0, 0, 0)
        
        roll = int(-9 + (roll * 2))
        pitch = int(-9 + (pitch * 2))
        self.scf.cf.commander.send_setpoint(roll, pitch, 0, thrust)
        time.sleep(0.01)

    def _get_state(self):
        """Retrieve the Crazyflie's current state (updated by the logging callback)."""
        return self.state

    def close(self):
        """Close the Crazyflie connection and stop logging."""
        self.log_conf.stop()
        self.scf.close_link()

class PhysicalRobotCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(PhysicalRobotCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self):
        # Save model and replay buffer periodically
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'model_steps_{self.n_calls}')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")
            
            # Save the replay buffer to disk
            replay_buffer_path = os.path.join(self.save_path, 'replay_buffer.pkl')
            with open(replay_buffer_path, 'wb') as f:
                pickle.dump(self.model.replay_buffer, f)
            if self.verbose > 0:
                print(f"Saving replay buffer to {replay_buffer_path}")

        return True

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
        position_reward = 0
        delx = abs(current_pos[0] - target_pos[0])
        dely = abs(current_pos[1] - target_pos[1])
        delz = abs(current_pos[2] - target_pos[2])
        position_reward -= (delz*20 + 5*delx + 5*dely)
        attitude = self.state[6:8]
        attitude_penalty = 0
        if delz > 0.2 :
            position_reward -= 20*delz
        if delx > 0.2:
            position_reward -= 30*delx
        if dely > 0.2:
            position_reward -= 30*dely
        if delx<0.1 and dely<0.1 and delz<0.1:
            position_reward += 25
        if delz == 0 :
            position_reward += 75

        self.cum_reward[0] -= delx
        self.cum_reward[1] -= dely
        self.cum_reward[2] -= delz
        
        velocities = self.state[3:6]
        velocity_penalty = -1.0 * np.linalg.norm(velocities)
        
        
        
        for angle in attitude:
            if angle > 60 or angle < -60: #make this 30 ig
                attitude_penalty -= 20

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
    
        # Send commands to drone
        thrust, roll, pitch, yaw = action
        n=1
        thrust_value = int(15000*np.exp(-0.001*self.step_count)+(thrust* 5000))
        thrust_value = min(65000, thrust_value)

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
    
    # Initialize callback
    callback = PhysicalRobotCallback(
        check_freq=1000,  # Save every 1000 steps
        save_path=save_dir,
        verbose=1
    )
    
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

    try:
        print("\nStarting training. Press Ctrl+C for emergency stop.")
        print("Ensure the physical space is clear and the drone is in the starting position.")
        input("Press Enter to begin...")

        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name="ppo_physical"
        )
        
        # Save final model and replay buffer
        model.save(f"{save_dir}/final_model")
        with open(f"{save_dir}/replay_buffer.pkl", 'wb') as f:
            pickle.dump(model.replay_buffer, f)
        print(f"Replay buffer saved to {save_dir}/replay_buffer.pkl")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving model...")
        model.save(f"{save_dir}/interrupted_model")
        with open(f"{save_dir}/replay_buffer.pkl", 'wb') as f:
            pickle.dump(model.replay_buffer, f)

    return model, env


def resume_training_with_replay_buffer(model_path, env, replay_buffer_path=None, total_timesteps=50000):
    """
    Resume training from a saved model and replay buffer
    """
    # Load the model
    model = PPO.load(model_path, env=env)

    if replay_buffer_path:
        # Load the replay buffer
        with open(replay_buffer_path, 'rb') as f:
            replay_buffer = pickle.load(f)
            model.replay_buffer = replay_buffer
            print(f"Loaded replay buffer from {replay_buffer_path}")
    
    # Now you can resume training with the loaded buffer
    save_dir = './cache/ppo_physical_crazyflie'
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
            callback=checkpoint_callback,
            tb_log_name="ppo_physical"
        )
        
        # Save final model and replay buffer
        model.save(f"{save_dir}/final_model")
        with open(f"{save_dir}/replay_buffer.pkl", 'wb') as f:
            pickle.dump(model.replay_buffer, f)

    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving model...")
        model.save(f"{save_dir}/interrupted_model")
        with open(f"{save_dir}/replay_buffer.pkl", 'wb') as f:
            pickle.dump(model.replay_buffer, f)

    return model, env


if __name__ == "__main__":
    # Check for saved model to resume training
    latest_model = None
    if len(sys.argv) > 1:
        latest_model = sys.argv[1]
        replay_buffer_path = None
        if len(sys.argv) > 2:
            replay_buffer_path = sys.argv[2]
        print(f"Resuming training from {latest_model}")
        env = PhysicalCrazyflieEnvWrapper()
        env = DummyVecEnv([lambda: env])
        model, env = resume_training_with_replay_buffer(latest_model, env, replay_buffer_path=replay_buffer_path)

    else:
        print("Starting new training session")
        model, env = train_ppo_physical()
