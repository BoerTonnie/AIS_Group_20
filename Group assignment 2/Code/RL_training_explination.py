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
        # Initialize communication with Arduino
        self.arduino = ArduinoCommunicator(port, baudrate, timeout)

        # Define the action space: single continuous action for servo angle
        self.action_space = Box(-1, 1, (1,), np.float32)

        # Define the observation space: distances, angles, and velocity data
        self.observation_space = Box(
            low=np.array([-1] * 13 + [-44]),    # Lower bounds for distances 0-9, angle, goal, distance to goal, and velocity
            high=np.array([1] * 13 + [44]),     # Upper bounds for distances, angle, goal, distance to goaland velocity
            dtype=np.float32
        )

        # Simulation parameters
        self.simulation_time = 0           #starting time
        self.delta_simulation_time = 0.05  # Time step in seconds
        self.max_simulation_time = 10      # Maximum simulation time in seconds
        self.goal_threshold = 0.2          # Threshold for reaching the goal

        # Distance and goal-related variables
        self.maxDistance = 1
        self.minDistance = -1
        self.distance = [0] * 10           # Initialize distances with zeros

        # Termination and stability counters
        self.steps_beyond_terminated = None
        self.stablecount = 0

        # Data storage for debugging and analysis
        self.data = pd.DataFrame(columns=['Time', 'Distance', 'Pitch', 'Goal', 'Reward'])

        # Set a random goal position within defined bounds
        self.goal = np.random.uniform(-0.7, 0.7)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation parameters
        self.simulation_time = 0
        self.goal = np.random.uniform(-0.7, 0.7)  # Reset goal position
        self.stablecount = 0

        # Reset servo angle to a neutral position via Arduino
        self.arduino.push_angle(0.05)
        time.sleep(0.3)  # Allow time for the servo to stabilize

        # Obtain initial distance and pitch readings
        self.distance[0], pitch = self.arduino.read_data()
        if self.distance[0] is None or pitch is None:
            self.distance[0], pitch = 0, 0  # Default values if no data received

        # Initialize distance history with the first reading
        for i in range(1, len(self.distance)):
            self.distance[i] = self.distance[0]

        # Reset termination condition and calculate initial velocity
        self.steps_beyond_terminated = None
        estimated_velocity = (self.distance[0] - self.distance[1]) / self.delta_simulation_time

        # Calculate the distance to the goal
        distanceToGoal = abs(self.goal) - abs(self.distance[0])

        # Construct the initial observation
        obs = np.array(self.distance + [pitch, self.goal, distanceToGoal, estimated_velocity], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Apply the action to the Arduino (servo angle adjustment)
        self.arduino.push_angle(action[0])
        time.sleep(0.02)  # Pause for hardware response
        self.simulation_time += self.delta_simulation_time  # Increment simulation time

        # Update distance history
        self.distance = [self.distance[i - 1] if i > 0 else 0 for i in range(len(self.distance))]

        # Read updated distance and pitch from Arduino
        self.distance[0], pitch = self.arduino.read_data()
        if self.distance[0] is None or pitch is None:
            self.distance[0], pitch = 0, 0  # Default values if no data received

        # Calculate the distance to the goal
        distanceToGoal = abs(self.goal) - abs(self.distance[0])

        # Check if the simulation time exceeds the maximum allowed
        terminated = self.simulation_time > self.max_simulation_time

        # Determine reward and check for goal stability
        if abs(self.distance[0] - self.goal) <= self.goal_threshold:
            self.stablecount += 1
        else:
            self.stablecount = 0

        # Calculate the reward based on action and proximity to the goal
        if not terminated:
            reward = -0.1 * abs(action[0])  # Penalize large actions
            reward += 1 * (1 - (abs(distanceToGoal) / self.maxDistance))  # Reward closeness to goal
            if self.goal - self.goal_threshold <= self.distance[0] <= self.goal + self.goal_threshold:
                reward += 0.5  # Bonus for reaching the goal
                if -0.1 <= pitch <= 0.2:
                    reward += 0.5  # Additional bonus for stability
                reward += self.stablecount
                if self.stablecount > 49:
                    terminated = True #terminates if stability reached
                    reward += 500  # Large reward for sustained stability
            if abs(self.distance[0]) > 0.95:  # Penalize edge positions
                reward -= 5
        elif self.steps_beyond_terminated is None: # if terminated is not init, init it and give small reward
            self.steps_beyond_terminated = 0
            reward = 0.5
        else:
            self.steps_beyond_terminated += 1  # if we are already terminated 0 reward
            reward = 0.0

        # Log data for debugging purposes
        self.data.loc[len(self.data)] = [
            self.simulation_time, self.distance[0], pitch, self.goal, reward
        ]

        # Calculate velocity for the next observation
        estimated_velocity = (self.distance[0] - self.distance[1]) / self.delta_simulation_time

        # Construct the observation
        obs = np.array(self.distance + [pitch, self.goal, distanceToGoal, estimated_velocity], dtype=np.float32)
        return obs, reward, terminated, False, {}

    def render(self):
        pass  # No rendering required for this environment

    def close(self):
        self.arduino.close()  # close serial connection with arduino so that it can be used for the next run or troubleshooting

    def save_data(self, filename="real_world_data.csv"):
        self.data.to_csv(filename, index=False)  # Save logged data to CSV
        print(f"Data saved to {filename}")

# Main training loop
if __name__ == "__main__":
    loadModel = True

    # Initialize the custom environment
    env = RealWorldEnv()

    # Wrap the environment for Stable-Baselines3 compatibility
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    # Define TensorBoard log directory
    tensorboard_log_dir = "tensorboard_logs/simulate_ppo"

    learningrate = 0.01

    if not loadModel:
        # Initialize a new PPO model
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=tensorboard_log_dir, learning_rate=learningrate)
    else:
        # Load an existing PPO model
        model = PPO.load("real_world_ppo_model", env=env, learningrate=learningrate)

    # Train the model with the specified number of timesteps
    model.learn(total_timesteps=20000)

    # Save the trained model
    model.save("real_world_ppo_model2")

    # Test the model
    obs, _ = env.reset()

