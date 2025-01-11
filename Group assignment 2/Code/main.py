import time
from control_serial import ArduinoCommunicator

def main():
    arduino = ArduinoCommunicator(port='COM7', baudrate=9600)
    try:
        while True:
            scale_dist, scale_pitch = arduino.read_data()
            if scale_dist is not None and scale_pitch is not None:
                print(f"Scaled Distance: {scale_dist:.3f},  Scaled Pitch: {scale_pitch:.3f}")
            arduino.push_angle(0.0)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Exiting script.")
    finally:
        arduino.close()

if __name__ == "__main__":
    main()
