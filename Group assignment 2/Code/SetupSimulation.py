# ------------- Imports ---------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random

from gymnasium.spaces import Box

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor



# ------------- Functions -------------

class simulate(gym.Env):
    def __init__(self):
        self.deltaSimulationTime = 0.05  # amount of time between each cycle of simulation
        self.speed = 0
        
        self.simulationTime = 0 

        self.maxDistance = 1
        self.minDistance = -1
        self.goal_threshold = 0.05
        self.noicePercentage = 0.0
        self.stablecount = 0
        self.stablegoal = 50

        self.estimated_velocity = 0

        # GENERATE RANDOM bounds
        self.lowerBound = self.minDistance + self.goal_threshold
        self.upperBound = self.maxDistance - self. goal_threshold

        randomDistance = random.uniform(self.lowerBound, self.upperBound)
        self.distance = [randomDistance] * 5
        self.goal = random.uniform(self.lowerBound, self.upperBound)
        self.angle = 0

        # calculate distance to goal
        self.distanceToGoal = self.goal - self.distance[0]

        self.maxSimulationTime = 150 # this number times 0.05 = simTime

        self.stepCounter = 0
        self.stepCounterMax = 10

        # Correctly initialize the DataFrame
        data = {
            'Time': [self.simulationTime],
            'Angle': [0],
            'Acceleration': [0],
            'Velocity': [self.speed],
            'Distance': [self.distance[0]],
            'Goal': [self.goal],
            'Distance To Goal': [self.distanceToGoal]
        }

        self.df = pd.DataFrame(data)
        self.action_space = Box(-1, 1, (1,), np.float32) # Action space for: Servo_Angle_Low, Servo_Angle_High, Shape(How much servos there are), data type
        # Define the observation space
        self.observation_space = Box(
            low=np.array([-1, -1, -1, -1, -1, -1, -1, -1, -44]),    # Lower bounds for distance[0:5], angle, goal, distance to goal
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 44]), # Upper bounds for distance[0:5], angle, goal, distance to goal
            dtype=np.float32
        )

        self.steps_beyond_terminated = None

        # Initialize a simulation cycle
        # `newCycle` requires an argument, so it can't be called here without fixing the input
        # Commenting out for now
        # self.newCycle()


    # Define the acceleration as a function of time, velocity, or position
    def acceleration(self, v, theta):
        return 9.81 * np.sin((2*theta)/np.pi)
    
    def mapAngle(self, servoAngle, direction):
        if direction == False:
            servo_min = -1
            servo_max = 1

            angle_min = -10
            angle_max = 10
        else:
            servo_min = -10
            servo_max = 10

            angle_min = -1
            angle_max = 1
            

        return (servoAngle - servo_min) * (angle_max - angle_min) / (servo_max - servo_min) + angle_min
    
    # Stores all info of the simulation for plotting purposes
    # Needs to be done after each itteration of the simulation
    def storeInfo(self, angle, acc):
        # Extract scalar values if these are numpy arrays
        speed_scalar = self.speed.item() if isinstance(self.speed, np.ndarray) else self.speed
        # distance_scalar = self.distance.item() if isinstance(self.distance, np.ndarray) else self.distance

        newData = {
            'Time': float(self.simulationTime),
            'Angle': float(angle) if isinstance(angle, (int, float)) else float(angle[0]),
            'Acceleration': float(acc) if isinstance(acc, (int, float)) else float(acc[0]),
            'Velocity': float(speed_scalar),
            'Distance': float(self.distance[0]),
            'Goal': float(self.goal),
            'Distance To Goal': float(self.distanceToGoal)
        }
        self.df.loc[len(self.df)] = newData
        
    def showDataframe(self):
        print(self.df)

    def plotDataframe(self):
        # Create subplots
        num_columns = len(self.df.columns) - 1  # Exclude 'Time' for x-axis
        fig, axes = plt.subplots(num_columns, 1, figsize=(8, 4 * num_columns), sharex=True)


        # Plot each column in a subplot
        for i, column in enumerate(self.df.columns[1:]):  # Skip 'Time' column
            axes[i].plot(self.df['Time'], self.df[column], label=column)
            axes[i].set_title(f'{column} vs Time')
            axes[i].set_ylabel(column)
            axes[i].grid(True)
            axes[i].legend()

        # Add x-axis label to the bottom plot
        axes[-1].set_xlabel('Time')

        # Adjust layout
        plt.tight_layout()
        plt.show()
    
    def saveDataframe(self, filename="Group assignment 2/Code/simulation_data.csv"):
        """
        Save the DataFrame to a CSV file.
        :param filename: Name of the CSV file to save.
        """
        self.df.to_csv(filename, index=False)
        print(f"Simulation data saved to {filename}")        

    def newCycle(self, IMUAngle):
        #print("New simulation cycle")
        # IMU angle: -1 - > 1
        # self.angle: -10 -> 10
        calculatedAngle = self.mapAngle(IMUAngle, direction=False)
        self.distance[4] = self.distance[3]
        self.distance[3] = self.distance[2]
        self.distance[2] = self.distance[1]
        self.distance[1] = self.distance[0]

        self.angle = IMUAngle
        acc = self.acceleration(self.speed, calculatedAngle)  # calculate acceleration on this cycle
        self.speed = self.speed + acc * self.deltaSimulationTime # update speed
        self.distance[0] = self.distance[0] + self.speed * self.deltaSimulationTime # update distance to sensor
        self.distance[0] = self.distance[0] * (1 - (random.uniform(self.lowerBound, self.upperBound) * self.noicePercentage))
        if self.distance[0] > 1.0: self.distance[0] = 1.0
        if self.distance[0] < -1.0: self.distance[0] = -1.0
        self.simulationTime = self.simulationTime + self.deltaSimulationTime # itterate time for data storing purposes

        # calculate distance to goal
        self.distanceToGoal = abs(self.goal) - abs(self.distance[0])

        # check for correct array configuration
        assert all(isinstance(d, float) for d in self.distance), "self.distance contains non-float elements!"

        # Store information for debugging and plotting purposes
        self.storeInfo(calculatedAngle, acc)




    def reset(self, *, seed: int | None = None, options: dict | None = None):

        if self.stepCounter == self.stepCounterMax:
            # Store simulation information
            self.saveDataframe()
            self.stepCounter = 0

        self.stepCounter += 1

        # reset super
        super().reset(seed=seed)

        # Reset simulation time to 0
        self.simulationTime = 0

        # set the servo angle
        self.angle = 0

        # GENERATE RANDOM TARGET
        lowerBound = self.minDistance + self.goal_threshold
        upperBound = self.maxDistance - self. goal_threshold

        # calculate distance to goal
        self.goal = random.uniform(lowerBound, upperBound)

        # read ball position and vel
        self.speed = 0

        randomDistance = random.uniform(self.lowerBound, self.upperBound)
        self.distance = [float(randomDistance)] * 5
        self.distanceToGoal = abs(self.goal) - abs(self.distance[0])

        # reset terminated condition
        self.steps_beyond_terminated = None

        # Reset dataframe to track data correctly
        data = {
            'Time': [self.simulationTime],
            'Angle': [0],
            'Acceleration': [0],
            'Velocity': [self.speed],
            'Distance': [self.distance[0]],
            'Goal': [self.goal],
            'Distance To Goal': [self.distanceToGoal]
        }
        self.df = pd.DataFrame(data)

        # check for correct array configuration
        assert all(isinstance(d, float) for d in self.distance), "self.distance contains non-float elements!"

        self.estimated_velocity = (self.distance[0] - self.distance[1]) / self.deltaSimulationTime

        # get obs
        obs = [self.distance[0], self.distance[1], self.distance[2], self.distance[3], self.distance[4], self.angle, self.goal, self.distanceToGoal, self.estimated_velocity]
        return np.array(obs, dtype=np.float32).flatten(), {} # might require extra info
    
    def step(self, action):
        # set the servo angle to action
        Actual_angle = action[0] # read actual (Should be -1 -> 1)

        # do simuloation stuff but later read from arduino
        self.newCycle(Actual_angle)
        # self.angle (-1 -> 1) represents -10 -> 10 degrees
        # self.speed (-1 -> 1) represents -10 -> 10 cm/s
        # self.distance (-1 -> 1) represemts 0 -> 30 cm



        # calculate termination
        terminated = bool(
            self.distance[0] < self.minDistance
            or self.distance[0] > self.maxDistance
            or self.simulationTime > self.maxSimulationTime
        )

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
            reward -= 0.1*Actual_angle
            if not(self.distance[0] < -0.95 or self.distance[0] > 0.95):    
                reward = float(1*(1 - (abs(self.distanceToGoal) / self.maxDistance)))
                if self.distance[0] > self.goal + self.goal_threshold and self.distance[0] < self.goal - self.goal_threshold:
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

        # check for correct array configuration
        assert all(isinstance(d, float) for d in self.distance), "self.distance contains non-float elements!"

        self.estimated_velocity = (self.distance[0] - self.distance[1]) / self.deltaSimulationTime

        # get obs
        obs = [self.distance[0], self.distance[1], self.distance[2], self.distance[3], self.distance[4], self.angle, self.goal, self.distanceToGoal, self.estimated_velocity]
        # print(f"obs: {obs}")
        return np.array(obs, dtype=np.float32).flatten(), reward, terminated, False, {}

    def render(self):
        pass

def plotDataframeFromCSV(filename):
    """
    Reads simulation data from a CSV file and plots the progress.

    :param filename: The path to the CSV file containing the simulation data.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filename)
        
        # Ensure the CSV contains a 'Time' column
        if 'Time' not in df.columns:
            raise ValueError("The CSV file must contain a 'Time' column.")

        # Create subplots for each column except 'Time'
        num_columns = len(df.columns) - 1  # Exclude 'Time'
        fig, axes = plt.subplots(num_columns, 1, figsize=(8, 4 * num_columns), sharex=True)

        # Plot each column against Time
        for i, column in enumerate(df.columns[1:]):  # Skip 'Time' column
            axes[i].plot(df['Time'], df[column], label=column)
            axes[i].set_title(f'{column} vs Time')
            axes[i].set_ylabel(column)
            axes[i].grid(True)
            axes[i].legend()

        # Add x-axis label to the bottom plot
        axes[-1].set_xlabel('Time')

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error while plotting from CSV: {e}")

# ------------- main cycle -------------

if __name__ == "__main__":
    # Initialize the environment
    env = simulate()

    # Wrap the environment with Monitor for logging
    env = Monitor(env)

    # Check if the environment adheres to the Gym API
    check_env(env, warn=True)

    # Wrap the environment for Stable-Baselines3
    vec_env = DummyVecEnv([lambda: env])

    # Directory for TensorBoard logs
    tensorboard_log_dir = "tensorboard_logs/simulate_ppo"

    # Initialize the RL model (PPO with MLP policy)
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=tensorboard_log_dir)

    # Train the model
    print("Model training started")
    model.learn(total_timesteps=10000)  # No custom callback

    # Save the trained model
    print("Training completed. Saving the trained model...")
    model.save("simulate_ppo_model")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")


    # Plot last saved simulation
    plotDataframeFromCSV("Group assignment 2\Code\simulation_data.csv")