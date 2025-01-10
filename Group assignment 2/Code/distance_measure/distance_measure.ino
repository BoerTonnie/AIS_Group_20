#include <Servo.h>
#include <Arduino_LSM6DS3.h>  // For the built-in IMU on Nano 33 IoT

// ===================== Pin Assignments ======================
const int sensorPin = A0;    // Sharp IR sensor output -> A0
const int servoPin  = 9;     // Servo control pin -> 9

Servo myServo;

// For serial input
String inputString = "";
bool stringComplete = false;

// ==========================================================================
// SETUP
// ==========================================================================
void setup() {
  Serial.begin(9600);         // Open serial at 9600 bps
  while (!Serial) {
    // Wait for Serial on Nano 33 IoT
  }

  // Servo setup
  myServo.attach(servoPin);   // Attach the servo to pin 9
  myServo.write(80);           // Optionally, set servo to 0 degrees

  // IMU setup
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1) {
      delay(10);
    }
  }

  // Intro messages
  Serial.println("=== Sharp Distance + Servo Control + IMU Angles ===");
  Serial.println("1) Distance from IR sensor on A0 will be printed.");
  Serial.println("2) Pitch and Roll angles from built-in IMU will be printed.");
  Serial.println("3) Type an angle (0-180) and press Enter to move the servo.");
  Serial.println("====================================================");
}

// ==========================================================================
// LOOP
// ==========================================================================
void loop() {
  // -----------------------------------------------------------------------
  // 1. Read distance from Sharp sensor
  // -----------------------------------------------------------------------
  float distanceCm = readSharpDistanceCm(sensorPin);

  // -----------------------------------------------------------------------
  // 2. Read IMU data (pitch & roll)
  // -----------------------------------------------------------------------
  float roll = 0, pitch = 0;
  float xAcc, yAcc, zAcc;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(xAcc, yAcc, zAcc);

    // Compute Roll (rotation around X-axis)
    //   roll  = atan2(Y-axis, Z-axis)
    roll = atan2(yAcc, zAcc) * 180.0 / PI;

    // Compute Pitch (rotation around Y-axis)
    //   pitch = atan2(-X-axis, sqrt(Y^2 + Z^2))
    pitch = atan2(-xAcc, sqrt(yAcc * yAcc + zAcc * zAcc)) * 180.0 / PI;
  }

  // -----------------------------------------------------------------------
  // 3. Print the Distance + Angles
  // -----------------------------------------------------------------------
  Serial.print("Distance: ");
  Serial.print(distanceCm, 1);
  Serial.print(" cm | Pitch: ");
  Serial.print(pitch, 1);
  Serial.println(" deg");

  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      // We got a newline -> user pressed Enter
      stringComplete = true;
    } else {
      // Accumulate the character into our inputString
      inputString += inChar;
    }
  }

  if (stringComplete) {
    int angle = inputString.toInt();      // Convert string to integer
    angle = constrain(angle, 24, 130);     // Constrain between 0 and 180
    myServo.write(angle);                 // Move servo
    inputString = "";
    stringComplete = false;
  }

  delay(20);
}

float readSharpDistanceCm(int pin) {
  int rawValue = analogRead(pin);                  // 0 - 1023
  float voltage = rawValue * (5.0 / 1023.0);       // Convert to voltage
  float distance = 27.728 * pow(voltage, -1.2045);
  if (distance < 0) {
    distance = 0;
  }
  return distance;
}
