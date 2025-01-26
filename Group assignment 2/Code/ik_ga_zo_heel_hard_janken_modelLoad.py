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
    def __init__(self, port='COM7', baudrate=115200, timeout=0.1):
        # Initialize Arduino communication
        self.arduino = ArduinoCommunicator(port, baudrate, timeout)

        # Gym environment setup
        self.action_space = Box(-1, 1, (1,), np.float32)
        # Define the observation space
        self.observation_space = Box(
            low=np.array([-1, -1, -1, -1, -1, -1, -1, -1, -44]),    # Lower bounds for distance[0:5], angle, goal, distance to goal
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 44]), # Upper bounds for distance[0:5], angle, goal, distance to goal
            dtype=np.float32
        )


        self.simulation_time = 0
        self.delta_simulation_time = 0.05  # time step in seconds
        self.max_simulation_time = 150  # max simulation time in seconds
        self.goal_threshold = 0.1

        self.maxDistance = 1
        self.minDistance = -1

        self.distance = [0, 0, 0, 0, 0]

        self.steps_beyond_terminated = None
        self.stablecount = 0

        # Data storage for debugging and analysis
        self.data = pd.DataFrame(columns=['Time', 'Distance', 'Pitch', 'Goal', 'Reward'])

        self.goal = np.random.uniform(-1, 1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation variables
        self.simulation_time = 0
        self.goal = np.random.uniform(-1, 1)
        self.stablecount = 0

        # Reset Arduino (if required)
        self.arduino.push_angle(0.05)  # Reset servo to neutral position
        time.sleep(0.1)


        # Initial observation from Arduino
        self.distance[0], pitch = self.arduino.read_data()
        if self.distance[0] is None or pitch is None:
            self.distance[0], pitch = 0, 0  # Default values if no data received
        self.distance[1] = self.distance[0]
        self.distance[2] = self.distance[0]
        self.distance[3] = self.distance[0]
        self.distance[4] = self.distance[0]

        # reset terminated condition
        self.steps_beyond_terminated = None

        # reset estimated velocity
        estimated_velocity = (self.distance[0] - self.distance[1]) / self.delta_simulation_time

        distanceToGoal = abs(self.goal) - abs(self.distance[0])

        print(self.goal)

        obs = np.array([self.distance[0], self.distance[1], self.distance[2], self.distance[3], self.distance[4], pitch, self.goal, distanceToGoal, estimated_velocity], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Send the action (servo angle) to the Arduino
        self.arduino.push_angle(action[0])
        time.sleep(0.04)
        # Wait for the next time step to complete
        self.simulation_time += self.delta_simulation_time

        # update distance memory
        self.distance[4] = self.distance[3]
        self.distance[3] = self.distance[2]
        self.distance[2] = self.distance[1]
        self.distance[1] = self.distance[0]

        # Read updated distance and pitch from the Arduino
        self.distance[0], pitch = self.arduino.read_data()
        if self.distance[0] is None or pitch is None:
            self.distance[0], pitch = 0, 0  # Default values if no data received

        distanceToGoal = abs(self.goal) - abs(self.distance[0])

        # Check termination conditions
        terminated = self.simulation_time > self.max_simulation_time or abs(self.distance[0]) > 1

        if abs(self.distance[0] - self.goal) < 0.05:
            self.stable_count += 1
            if self.stable_count >= 50:
                terminated = True
                reward += 10  # Bonus for achieving stability
        else:
            self.stable_count = 0
        



# ----------- reward ------------------
        if abs(self.distance[0] - self.goal) <= self.goal_threshold:
            self.stablecount += 1
        else:
            self.stablecount = 0

        if self.stablecount > 50:
                        terminated = True

        # calcualte reward
        # Question to Hussam: Should we also put the goal somewhere else so the model knows where to aim itself to?
        if not terminated:
            reward = float(0)
            reward -= 0.01*abs(action[0]) # punih based on the angle 
            reward += float(1*(1 - (abs(distanceToGoal) / self.maxDistance))) # reward proportional to the distance to goal
            if self.distance[0] < self.goal + self.goal_threshold and self.distance[0] > self.goal - self.goal_threshold:
                reward += 0.5
                if self.stablecount > 49:
                    reward += 500
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = 0.5
        else:
            if self.steps_beyond_terminated == 0:
                print(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0        
# ----------- end reward ------------------


        # Log data for debugging
        self.data.loc[len(self.data)] = [
            self.simulation_time, self.distance[0], pitch, self.goal, reward
        ]

        

        # reset estimated velocity
        estimated_velocity = (self.distance[0] - self.distance[1]) / self.delta_simulation_time

        # Construct observation
        obs = np.array([self.distance[0], self.distance[1], self.distance[2], self.distance[3], self.distance[4], pitch, self.goal, distanceToGoal, estimated_velocity], dtype=np.float32)
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
    loadModel = False

    # Initialize environment
    env = RealWorldEnv()

    # # Wrap the environment for Stable-Baselines3
    # env = Monitor(env)
    # vec_env = DummyVecEnv([lambda: env])

    # Load old model
    model = PPO.load("real_world_ppo_modelD", env=env)

    obs = env.reset()

    # Execute the model without further training
    for _ in range(1000):  # Run for 1000 steps or adjust as needed
        env.render()  # Render the environment (optional)
        
        # Predict the action without learning
        action, _states = model.predict(obs, deterministic=True)  # deterministic=True ensures consistent behavior
        
        # Take the action in the environment
        obs, reward, done, info = env.step(action)
        
        # Reset the environment if done
        if done:
            obs = env.reset()

    # Test the model
    obs, _ = env.reset()
