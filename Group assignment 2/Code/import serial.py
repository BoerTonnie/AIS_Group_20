#import the libraries for reading the serial port, wait functions and decoding asci
import serial
import time
import re

def main():
    # Open the serial port
    ser = serial.Serial(port='COM7', baudrate=9600, timeout=0.1)
    # Give the board a moment to reset after opening the port
    time.sleep(1)
    print("Connected to", ser.port)


    # Regex patterns to parse lines like:
    # "Distance: 12.3 cm | Pitch: -5.0 deg"
    distance_pattern = re.compile(r"Distance:\s*([\d\.]+)\s*cm")
    pitch_pattern = re.compile(r"Pitch:\s*([\-\d\.]+)\s*deg")
    #start angle
    input_angle = 0

    try:
        while True:
            line = None
            while ser.in_waiting > 0:
                raw_line =ser.readline()
                line = raw_line.decode('utf-8', errors='ignore').strip()
            if line is not None:
                # Parse distance, pitch, roll (if present)
                dist_match = distance_pattern.search(line)
                pitch_match = pitch_pattern.search(line)
                if dist_match:
                    dist_val = float(dist_match.group(1))
                if pitch_match:
                    pitch_val = float(pitch_match.group(1))

                scale_dist = scale(dist_val, 4, 20, -1, 1)
                scale_pitch = scale(pitch_val, -10, 13, -1, 1)
                print(dist_val,pitch_val,scale_dist,scale_pitch)    

            current_angle = scale(input_angle, -1, 1, 24, 130)
            # Constrain angle between 24 and 120
            angle_val = max(24, min(130, current_angle))
            ser.write((str(angle_val) + "\n").encode('utf-8'))
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting script.")
    finally:
        ser.close()

def scale(value, input_min, input_max, output_min, output_max):
    # Scale the input value to the output range
    scale_factor = (output_max - output_min) / (input_max - input_min)
    out = output_min + (value - input_min) * scale_factor
    if out>output_max:
        out=output_max
    if out<output_min:
        out=output_min
    return out


if __name__ == "__main__":
    main()
