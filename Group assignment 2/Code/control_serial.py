import serial
import time
import re

class ArduinoCommunicator:
    def __init__(self, port='COM7', baudrate=11500, timeout=0.1):
        """
        Initialize the serial connection and compile regex patterns.
        """
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        time.sleep(1)  # Allow the board to reset if needed

        print(f"Connected to {self.ser.port}")

        # Compile your regex patterns here
        # For lines like "D0.98  P-0.01"
        self.distance_pattern = re.compile(r"D([\-\d\.]+)")
        self.pitch_pattern    = re.compile(r"P([\-\d\.]+)")
        self.lastdist = 0.0
        self.lastpitch = 0.0


    def read_data(self):
        """
        Read all available lines from the serial buffer and keep only the most recent.
        Parse the line for distance and pitch, then apply scaling.
        
        Returns:
            (scale_dist, scale_pitch) as floats, or (None, None) if nothing read.
        """
        line = None
        # Drain the serial buffer
        while self.ser.in_waiting > 0:
            raw_line = self.ser.readline()
            line = raw_line.decode('utf-8', errors='ignore').strip()
            
        # If we got a new line, parse it
        if line is not None:
            # Print (optional) for debugging
            # print("Raw line from Arduino:", line)

            dist_val = None
            pitch_val = None

            # Extract distance
            dist_match = self.distance_pattern.search(line)
            if dist_match:
                dist_val = float(dist_match.group(1))

            # Extract pitch
            pitch_match = self.pitch_pattern.search(line)
            if pitch_match:
                pitch_val = float(pitch_match.group(1))

            # Scale if we have valid values
            # (You can decide how to handle if dist_val or pitch_val is None)
            if dist_val is not None:
                dist = self._scale(dist_val, -1, 1, -1, 1)
            else:
                dist = None

            if pitch_val is not None:
                pitch = self._scale(pitch_val, -1, 1, -1, 1)
            else:
                pitch = None
            self.lastdist = dist
            self.lastpitch = pitch
            return dist, pitch

        # If no line was available, return None
        return self.lastdist, self.lastpitch

    def push_angle(self, input_angle):
        """
        Takes an input angle in some range (e.g. -1 to +1),
        scales it to 24..130, and sends it to the Arduino.

        Example usage:
            push_angle(0.0)    ->  sends 77   (approx mid-range)
            push_angle(1.0)    ->  sends 130
            push_angle(-1.0)   ->  sends 24
        """
        # Scale angle
        current_angle = self._scale(input_angle, -1, 1, 24, 130)
        # Constrain between 24 and 130
        angle_val = max(50, min(140, current_angle))

        # Send over serial
        self.ser.write((str(angle_val) + "\n").encode('utf-8'))

        # (Optional) print debugging
        # print(f"Sent angle: {angle_val}")

    def close(self):
        """
        Close the serial connection.
        """
        self.ser.close()
        print("Serial connection closed.")

    def _scale(self, value, input_min, input_max, output_min, output_max):
        """
        Scale 'value' from input range [input_min..input_max]
        to output range [output_min..output_max], then clamp.
        """
        scale_factor = (output_max - output_min) / (input_max - input_min)
        out = output_min + (value - input_min) * scale_factor
        if out > output_max:
            out = output_max
        if out < output_min:
            out = output_min
        return out
