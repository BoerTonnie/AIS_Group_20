# ------------- Imports ---------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# ------------- Variables & constants -------------

simulatedCycleTime_ms = 50

gravity = 9.81
startingAngle = 0
startingDistance = 100

ballRadius = 12.5

rail_length = 250 # length of the rail, accessible to the ball
sensor_Offset = 40 # distance from sensor to the rail



ServoAngle = 0 # placeholder for the eventual input value from the model






# ------------- Functions -------------

class simulate:
    def __init__(self, StartingDistance):
        self.velocity = 0
        self.speed = 0
        self.distance = StartingDistance

        self.newCycle()

    # Define the acceleration as a function of time, velocity, or position
    def acceleration(t, v, x, theta):
        return -9.8 * np.sin((2*theta)/np.pi) + 0.1 * v  #  gravity * sin(angle) + velocity-dependent drag
    
    def mapAngle(ServoAngle):
        servo_min = 0
        servo_max = 255

        angle_min = -10
        angle_max = 10

        return (ServoAngle - servo_min) * (angle_max - angle_min) / (servo_max - servo_min) + angle_min
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        

        

    def newCycle():
        #newDistance = self.distance * 
        print("New simulation cycle")

        angle = mapAngle(ServoAngle)

        # Define the acceleration as a function of time, velocity, or position
    def acceleration(t, v, x):
        return -9.8 + 0.1 * v  # Example: gravity + velocity-dependent drag

    def oops(theta):


        # Define the acceleration as a function of time, velocity, or position
        def acceleration(t, v, x, theta):
            return -9.8 * np.sin((2*theta)/np.pi) + 0.1 * v  # Example: gravity + velocity-dependent drag

        # Simulation parameters
        dt = 0.01  # Time step (seconds)
        t_end = 20  # End time (seconds)

        # Initial conditions
        x = 0.0  # Initial position (meters)
        v = 20.0  # Initial velocity (meters/second)
        t = 0.0   # Initial time (seconds)

        # Lists to store the results for plotting
        time = [t]
        position = [x]
        velocity = [v]

        # Simulation loop
        while t < t_end:
            a = acceleration(t, v, x, theta)  # Calculate acceleration
            v = v + a * dt             # Update velocity
            x = x + v * dt             # Update position
            t = t + dt                 # Update time

            # Store results
            time.append(t)
            position.append(x)
            velocity.append(v)

        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(time, position, label="Position (x)")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(time, velocity, label="Velocity (v)", color="orange")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.legend()

        plt.tight_layout()

        print (position)
        plt.show()







# ------------- main cycle -------------

if __name__ == "__main__":
    simulation = simulate
    theta = 10
    simulation.oops(theta)


    print(50)