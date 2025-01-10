# ------------- Imports ---------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.spaces import Box




# ------------- Variables & constants -------------

# simulatedCycleTime_ms = 50

# gravity = 9.81
# startingAngle = 0
# startingDistance = 100

# ballRadius = 12.5

# rail_length = 250 # length of the rail, accessible to the ball
# sensor_Offset = 40 # distance from sensor to the rail




# ------------- Functions -------------

class simulate(gym.Env):
    def __init__(self, StartingDistance):
        self.deltaSimulationTime = 0.05  # amount of time between each cycle of simulation
        self.speed = 0
        self.distance = StartingDistance
        self.simulationTime = 0

        # Correctly initialize the DataFrame
        data = {
            'Time': [self.simulationTime],
            'Angle': [0],
            'Acceleration': [0],
            'Velocity': [self.speed],
            'Distance': [self.distance]
        }

        self.df = pd.DataFrame(data)
        self.action_space = Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (1,), np.float32)
        # Define the observation space
        self.observation_space = Box(
            low=np.array([0.0, -10, -10]),    # Lower bounds for distance, velocity, and angle
            high=np.array([30, 10, 10]), # Upper bounds for distance, velocity, and angle
            dtype=np.float32
        )

        # Initialize a simulation cycle
        # `newCycle` requires an argument, so it can't be called here without fixing the input
        # Commenting out for now
        # self.newCycle()


    # Define the acceleration as a function of time, velocity, or position
    def acceleration(self, v, theta):
        return -9.81 * np.sin((2*theta)/np.pi)
    
    def mapAngle(self, servoAngle):
        servo_min = 60
        servo_max = 100

        angle_min = -10
        angle_max = 10

        return (servoAngle - servo_min) * (angle_max - angle_min) / (servo_max - servo_min) + angle_min
    
    # Stores all info of the simulation for plotting purposes
    # Needs to be done after each itteration of the simulation
    def storeInfo(self, angle, acc):
        newData = {
            'Time': self.simulationTime,
            'Angle': angle,
            'Acceleration': acc,
            'Velocity': self.speed,
            'Distance': self.distance
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

        angle = self.mapAngle(servoAngle)
        acc = self.acceleration(self.speed, angle)  # calculate acceleration on this cycle
        self.speed = self.speed + acc * self.deltaSimulationTime # update speed
        self.distance = self.distance + self.speed * self.deltaSimulationTime # update distance to sensor
        self.simulationTime = self.simulationTime + self.deltaSimulationTime # itterate time for data storing purposes

        # Store information for debugging and plotting purposes
        self.storeInfo(angle, acc)




    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # reset super
        # set the servo angle
        # GENERATE RANDOM TARGET
        #
        # read ball position and vel
        # return obs
        pass

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

    sim = simulate(10)
    print (sim.observation_space.sample())