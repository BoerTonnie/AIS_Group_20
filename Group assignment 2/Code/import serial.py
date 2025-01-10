import serial
import time
import re

def main():
    # ----------------------------------------------------------------------
    # 1. Open Serial Port
    #    Change 'COM3' to whatever port your Arduino is on (Windows example).
    #    On Linux/Mac, it might be '/dev/ttyACM0' or '/dev/ttyUSB0'.
    # ----------------------------------------------------------------------
    ser = serial.Serial(port='COM7', baudrate=9600, timeout=0.1)

    # Give the board a second to reset after opening the port
    time.sleep(2)

    print("Connected to", ser.port)
    print("Type an angle (0-180) and press Enter to send to servo.")
    print("Press Ctrl+C to exit.")
    print("-------------------------------------------------------")

    # Pattern to match lines of the form:
    # Distance: 12.3 cm | Roll: 2.3 deg | Pitch: -5.0 deg
    # We'll use a simple regex to extract numeric values for Distance and Pitch.
    distance_pattern = re.compile(r"Distance:\s*([\d\.]+)\s*cm")
    pitch_pattern = re.compile(r"Pitch:\s*([\-\d\.]+)\s*deg")

    try:
        while True:
            # ------------------------------------------------------------------
            # 2a. Check if there's any new data from Arduino
            #     We'll read a line if available and decode it.
            # ------------------------------------------------------------------
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()

                # Print the raw line (for debugging)
                print("Arduino says:", line)

                # If you want to parse distance and pitch from the line:
                dist_match = distance_pattern.search(line)
                pitch_match = pitch_pattern.search(line)

                if dist_match:
                    dist_val = float(dist_match.group(1))
                    # Do something with dist_val if needed
                    # e.g., print(f"Parsed distance: {dist_val} cm")

                if pitch_match:
                    pitch_val = float(pitch_match.group(1))
                    # Do something with pitch_val if needed
                    # e.g., print(f"Parsed pitch: {pitch_val} deg")

            # ------------------------------------------------------------------
            # 2b. Check if user typed something in Python console
            #     We'll do a non-blocking approach using input() in a try/except
            #     or you could do it a different way. For demonstration,
            #     let's do a quick hack: attempt to read from stdin quickly.
            # ------------------------------------------------------------------
            angle_cmd = input_non_blocking()
            if angle_cmd is not None:
                # Clean up input (remove spaces, etc.)
                angle_cmd = angle_cmd.strip()
                if angle_cmd.isdigit():
                    angle_val = int(angle_cmd)
                    # Constrain between 0 and 180
                    angle_val = max(0, min(180, angle_val))
                    # Send to Arduino (Arduino expects a line with just the angle)
                    ser.write((str(angle_val) + "\n").encode('utf-8'))
                    print(f"Sent servo angle: {angle_val}")
                else:
                    print("Please enter a valid integer for servo angle (0-180).")

            # Small delay so we don’t thrash the CPU
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nExiting script.")
    finally:
        ser.close()

def input_non_blocking():
    """
    Attempt to read a line from stdin without blocking.
    If nothing is there, return None.
    If something is there, return the string.
    
    This approach uses the 'select' module on Unix-like systems.
    On Windows, it might not work as intended. For a fully 
    cross-platform non-blocking input solution, you may need a 
    different approach or a library like 'pyreadline'.
    """
    import sys, select
    # Check if there's data available on stdin
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline()
    return None

if __name__ == "__main__":
    main()
