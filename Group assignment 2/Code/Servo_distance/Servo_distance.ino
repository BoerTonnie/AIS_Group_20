#include <Servo.h>

// ===================== Pin Assignments ======================
const int sensorPin = A0;    // Sharp IR sensor output -> A0
const int servoPin  = 9;     // Servo control pin -> 9

Servo myServo;

// For serial input
String inputString = "";
bool stringComplete = false;

void setup() {
  Serial.begin(9600);        // Open serial at 9600 bps
  myServo.attach(servoPin);  // Attach the servo to pin 9
  myServo.write(0);          // Optionally, set servo to 0 degrees

  Serial.println("=== Sharp Distance + Servo Control ===");
  Serial.println("1) Distance from IR sensor on A0 will be printed.");
  Serial.println("2) Type an angle (0-180) and press Enter to move the servo.");
  Serial.println("========================================");
}

void loop() {
  // -----------------------------------------------------------------------
  // 1. Read distance from Sharp sensor and print it
  // -----------------------------------------------------------------------
  float distanceCm = readSharpDistanceCm(sensorPin);
  Serial.print("Distance: ");
  Serial.print(distanceCm, 1);  // Print one decimal place
  Serial.println(" cm");

  // -----------------------------------------------------------------------
  // 2. Check for any new serial data (angle commands)
  // -----------------------------------------------------------------------
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

  // -----------------------------------------------------------------------
  // 3. If a complete command was received, parse and set the servo angle
  // -----------------------------------------------------------------------
  if (stringComplete) {
    int angle = inputString.toInt();            // Convert string to integer
    angle = constrain(angle, 0, 180);           // Constrain between 0 and 180

    myServo.write(angle);                       // Move servo
    Serial.print(angle);
   

    // Clear for next time
    inputString = "";
    stringComplete = false;
  }

  // -----------------------------------------------------------------------
  // 4. (Optional) Delay for readability (tune as needed)
  // -----------------------------------------------------------------------
  delay(200);
}

// ==========================================================================
// Function: Reads Sharp GP2Y0A21YK0F distance sensor in cm
// ==========================================================================
float readSharpDistanceCm(int pin) {
  // 0 - 1023
  int rawValue = analogRead(pin);

  // Convert to voltage (assuming 5.0 V reference)
  float voltage = rawValue * (5.0 / 1023.0);

  // Approximate formula for GP2Y0A21YK0F
  // distance (cm) = 27.728 * (voltage ^ -1.2045)
  // Adjust coefficients if you perform your own calibration
  float distance = 27.728 * pow(voltage, -1.2045);

  // Prevent negative values in case of sensor anomalies
  if (distance < 0) {
    distance = 0;
  }

  return distance;
}
