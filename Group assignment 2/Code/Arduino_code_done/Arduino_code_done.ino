#include <Servo.h>
#include <Arduino_LSM6DS3.h>  // For the built-in IMU on Nano 33 IoT

const int sensorPin = A0;    // Sharp IR sensor output -> A0
const int servoPin  = 9;     // Servo control pin -> 9
float scdistance = 0.0;
float scpitch = 0.0;

const uint8_t NUM_SAMPLES = 10;
int samples[NUM_SAMPLES];
long runningSum = 0;
uint8_t currentIndex = 0;

Servo myServo;
String inputString = "";
bool stringComplete = false;

unsigned long previousMillis = 0;
const long interval = 5; // 10 milliseconds for 100 times per second

void setup() {
  Serial.begin(115200);
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
  for (uint8_t i = 0; i < NUM_SAMPLES; i++) {
    samples[i] = 0;
    runningSum += samples[i];
  }
}

void loop() {
  //init some var's
  float distanceCm = readSharpDistanceCm(sensorPin);
  float roll = 0, pitch = 0;
  float xAcc, yAcc, zAcc;
  float scdistance = 0;
  float scpitch = 0;

  //reading the acclerometer
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(xAcc, yAcc, zAcc);
    pitch = atan2(-xAcc, sqrt(yAcc * yAcc + zAcc * zAcc)) * 180.0 / PI;
  }

  //read the serial port
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }

  //convert string from serial to int
  if (stringComplete) {
    int angle = inputString.toInt();      // Convert string to integer
    angle = constrain(angle, 24, 120);     // Constrain between 0 and 180
    myServo.write(angle);                 // Move servo
    // Clear for next time
    inputString = "";
    stringComplete = false;
  }

  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    // Save the last time you ran the code
    previousMillis = currentMillis;

    // map and print distance and pitch to -1 to 1
    scdistance = map(distanceCm*100, 400, 1800, -1000, 1000);
    scpitch = map(pitch*100, -1000, 1100, -1000, 1000);
    Serial.print(scdistance/1000, 2);
    Serial.print("  ");
    Serial.println(scpitch/1000, 2);
  }
}

float readSharpDistanceCm(int pin) {
  runningSum -= samples[currentIndex];
  int rawValue = analogRead(pin);                  // 0 - 1023
  samples[currentIndex] = rawValue;
  runningSum += rawValue;
  currentIndex++;
  if (currentIndex >= NUM_SAMPLES) {
    currentIndex = 0;
  }
  int average = runningSum / NUM_SAMPLES;
  float voltage = average * (5.0 / 1023.0);       // Convert to voltage
  float distance = 27.728 * pow(voltage, -1.2045);
  if (distance < 0) {
    distance = 0;
  }
  return distance;
}
