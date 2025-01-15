import numpy as np
import pandas as pd
import time
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from control_serial import ArduinoCommunicator


class RealWorldEnv(Env):
    def __init__(self, port='COM3', baudrate=115200, timeout=0.1):
        # Initialize Arduino communication
        self.arduino = ArduinoCommunicator(port, baudrate, timeout)

        # Gym environment setup
        self.action_space = Box(-1, 1, (1,), np.float32)
        self.observation_space = Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )

        self.simulation_time = 0
        self.delta_simulation_time = 0.05  # time step in seconds
        self.max_simulation_time = 150  # max simulation time in seconds

        # Data storage for debugging and analysis
        self.data = pd.DataFrame(columns=['Time', 'Distance', 'Pitch', 'Goal', 'Reward'])
        self.goal = np.random.uniform(-1, 1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation variables
        self.simulation_time = 0
        self.goal = np.random.uniform(-1, 1)
        self.stable_count = 0

        # Reset Arduino (if required)
        self.arduino.push_angle(0)  # Reset servo to neutral position

        # Initial observation from Arduino
        distance, pitch = self.arduino.read_data()
        if distance is None or pitch is None:
            distance, pitch = 0, 0  # Default values if no data received

        obs = np.array([distance, pitch, self.goal], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Send the action (servo angle) to the Arduino
        self.arduino.push_angle(action[0])
        time.sleep(0.01)
        # Wait for the next time step to complete
        self.simulation_time += self.delta_simulation_time

        # Read updated distance and pitch from the Arduino
        distance, pitch = self.arduino.read_data()
        if distance is None or pitch is None:
            distance, pitch = 0, 0  # Default values if no data received

        # Calculate distance to goal and reward
        distance_to_goal = abs(self.goal - distance)
        reward = 1 - (distance_to_goal / 2)  # Reward based on proximity to goal

        # Check termination conditions
        terminated = self.simulation_time > self.max_simulation_time or abs(distance) > 1

        if abs(distance - self.goal) < 0.05:
            self.stable_count += 1
            if self.stable_count >= 50:
                terminated = True
                reward += 10  # Bonus for achieving stability
        else:
            self.stable_count = 0

        # Log data for debugging
        self.data.loc[len(self.data)] = [
            self.simulation_time, distance, pitch, self.goal, reward
        ]

        # Construct observation
        obs = np.array([distance, pitch, self.goal], dtype=np.float32)
        return obs, reward, terminated, False, {}

    def render(self):
        pass

    def close(self):
        self.arduino.close()

    def save_data(self, filename="real_world_data.csv"):
        self.data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

# Main training loop
if __name__ == "__main__":
    # Initialize environment
    env = RealWorldEnv()

    # Wrap the environment for Stable-Baselines3
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    # Directory for TensorBoard logs
    tensorboard_log_dir = "tensorboard_logs/simulate_ppo"



    # Initialize RL model
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=tensorboard_log_dir)

    # Train the model
    model.learn(total_timesteps=10000)

    # Save the model
    model.save("real_world_ppo_model")

    # Test the model
    obs, _ = env.reset()
