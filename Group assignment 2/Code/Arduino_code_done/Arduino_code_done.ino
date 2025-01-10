#include <Servo.h>
#include <Arduino_LSM6DS3.h>  // For the built-in IMU on Nano 33 IoT

const int sensorPin = A0;    // Sharp IR sensor output -> A0
const int servoPin  = 9;     // Servo control pin -> 9
float scdistance = 0.0;
float scpitch = 0.0;


Servo myServo;
String inputString = "";
bool stringComplete = false;

void setup() {
  Serial.begin(9600);
  while (!Serial) {
  }

  // Servo setup
  myServo.attach(servoPin);
  myServo.write(80);

  // IMU setup
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1) {
      delay(10);
    }
  }
}

void loop() {
  float distanceCm = readSharpDistanceCm(sensorPin);
  float roll = 0, pitch = 0;
  float xAcc, yAcc, zAcc;
  float scdistance = 0;
  float scpitch = 0;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(xAcc, yAcc, zAcc);
    pitch = atan2(-xAcc, sqrt(yAcc * yAcc + zAcc * zAcc)) * 180.0 / PI;
  }
  scdistance = map(distanceCm*100, 400, 1800, -1000, 1000);
  scpitch = map(pitch*100, -1000, 1100, -1000, 1000);

  Serial.print(scdistance/1000, 2);
  Serial.print("  ");
  Serial.println(scpitch/1000, 2);

  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }

  if (stringComplete) {
    int angle = inputString.toInt();      // Convert string to integer
    angle = constrain(angle, 24, 120);     // Constrain between 0 and 180
    myServo.write(angle);                 // Move servo
    // Clear for next time
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
