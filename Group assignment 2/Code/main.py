import time
from control_serial import ArduinoCommunicator

def main():
    arduino = ArduinoCommunicator(port='COM3', baudrate=115200)
    try:
        while True:
            dist, pitch = arduino.read_data()
            if dist is not None and pitch is not None:
                print(f"Distance: {dist:.2f},  Pitch: {pitch:.2f}")
            arduino.push_angle(0.0)
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Exiting script.")
    finally:
        arduino.close()

if __name__ == "__main__":
    main()
