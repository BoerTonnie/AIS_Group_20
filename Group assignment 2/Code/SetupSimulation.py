# ------------- Imports ---------------
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



# ------------- Functions -------------

class simulate(gym.Env):
    def __init__(self, StartingDistance):
        self.deltaSimulationTime = 0.05  # amount of time between each cycle of simulation
        self.speed = 0
        self.distance = StartingDistance
        self.simulationTime = 0

        self.maxDistance = 30
        self.minDistance = 4
        self.goal_threshold = 0.5

        # Correctly initialize the DataFrame
        data = {
            'Time': [self.simulationTime],
            'Angle': [0],
            'Acceleration': [0],
            'Velocity': [self.speed],
            'Distance': [self.distance]
        }

        self.df = pd.DataFrame(data)
        self.action_space = Box(-1, 1, (1,), np.float32) # Action space for: Servo_Angle_Low, Servo_Angle_High, Shape(How much servos there are), data type
        # Define the observation space
        self.observation_space = Box(
            low=np.array([0.0, -10, -10]),    # Lower bounds for distance, velocity, and angle
            high=np.array([30, 10, 10]), # Upper bounds for distance, velocity, and angle
            dtype=np.float32
        )

        self.steps_beyond_terminated = None

        # Initialize a simulation cycle
        # `newCycle` requires an argument, so it can't be called here without fixing the input
        # Commenting out for now
        # self.newCycle()


    # Define the acceleration as a function of time, velocity, or position
    def acceleration(self, v, theta):
        return -9.81 * np.sin((2*theta)/np.pi)
    
    def mapAngle(self, servoAngle):
        servo_min = -1
        servo_max = 1

        angle_min = -10
        angle_max = 10

        return (servoAngle - servo_min) * (angle_max - angle_min) / (servo_max - servo_min) + angle_min
    
    # Stores all info of the simulation for plotting purposes
    # Needs to be done after each itteration of the simulation
    def storeInfo(self, angle, acc):
        # Extract scalar values if these are numpy arrays
        speed_scalar = self.speed.item() if isinstance(self.speed, np.ndarray) else self.speed
        distance_scalar = self.distance.item() if isinstance(self.distance, np.ndarray) else self.distance

        newData = {
            'Time': float(self.simulationTime),
            'Angle': float(angle) if isinstance(angle, (int, float)) else float(angle[0]),
            'Acceleration': float(acc) if isinstance(acc, (int, float)) else float(acc[0]),
            'Velocity': float(speed_scalar),
            'Distance': float(distance_scalar)
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

        

    def newCycle(self, servoAngle):
        #newDistance = self.distance * 
        #print("New simulation cycle")

        self.angle = self.mapAngle(servoAngle)
        acc = self.acceleration(self.speed, self.angle)  # calculate acceleration on this cycle
        self.speed = self.speed + acc * self.deltaSimulationTime # update speed
        self.distance = self.distance + self.speed * self.deltaSimulationTime # update distance to sensor
        self.simulationTime = self.simulationTime + self.deltaSimulationTime # itterate time for data storing purposes

        # Store information for debugging and plotting purposes
        self.storeInfo(self.angle, acc)




    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # reset super
        super().reset(seed=seed)

        # set the servo angle
        self.angle = 0

        # GENERATE RANDOM TARGET
        lowerBound = self.minDistance + self.goal_threshold
        upperBound = self.maxDistance - self. goal_threshold
        self.goal = random.uniform(lowerBound, upperBound)


        # read ball position and vel
        self.speed = 0 # read actual

        self.distance = random.uniform(lowerBound, upperBound) # read actual 

        # reset terminated condition
        self.steps_beyond_terminated = None

        # return obs
        obs = [self.distance, self.speed, self.angle] # check if extran info is needed in stable baseline 3
        return np.array(obs, dtype=np.float32).flatten(), {} # might require extra info
    



    def step(self, action):
        # set the servo angle to action
        Actual_angle = action # read actual

        # do simuloation stuff but later read from arduino
        self.newCycle(Actual_angle)

        # calculate termination
        terminated = bool(
            self.distance < self.minDistance
            or self.distance > self.maxDistance
        )

        # calcualte reward
        # Question to Hussam: Should we also put the goal somewhere else so the model knows where to aim itself to?
        if not terminated:
            reward = 0.5
            if self.distance > self.goal + self.goal_threshold and self.distance < self.goal - self.goal_threshold:
                reward += 0.5
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

        # get obs
        obs = [self.distance, self.speed, self.angle]
        return np.array(obs, dtype=np.float32).flatten(), reward, terminated, False, {}

    def render(self):
        pass

# ------------- main cycle -------------

if __name__ == "__main__":
    # sim = simulate(10)
    # theta = 10
    # simLoops = 100
    # for i in range(simLoops):
    #     sim.newCycle(theta)
    # sim.showDataframe()
    # sim.plotDataframe()

    # sim = simulate(10)
    # print (sim.observation_space.sample())

    # base_env = sim.make("FliFlaFloe")
    # print (base_env.action_space)



    # Initialize the environment with a starting distance
    env = simulate(StartingDistance=10)

    # check if the environment adheres to the Gym API
    check_env(env, warn=True)

    # wrap the environment for Stable-Baselines3
    vec_env = DummyVecEnv([lambda: env])

    # initialize the RL model (PPO with MLP policy)
    model = PPO("MlpPolicy", vec_env, verbose=1)

    # train the model
    print("model train start")
    model.learn(total_timesteps=10000)

    # Save the trained model
    print("Training completed \n\n\nSave the trained model")
    model.save("simulate_ppo_model")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Test the trained model
    obs, _ = env.reset()

    for _ in range (200): # Run for 200 steps
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset