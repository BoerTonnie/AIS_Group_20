import time
from control_serial import ArduinoCommunicator

def main():
    arduino = ArduinoCommunicator(port='COM7', baudrate=9600)
    try:
        while True:
            dist, pitch = arduino.read_data()
            if dist is not None and pitch is not None:
                print(f"Distance: {dist:.3f},  Pitch: {pitch:.3f}")
            arduino.push_angle(0.0)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Exiting script.")
    finally:
        arduino.close()

if __name__ == "__main__":
    main()
