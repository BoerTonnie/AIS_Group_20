#include <Servo.h>
#include <Arduino_LSM6DS3.h>  // For the built-in IMU on Nano 33 IoT

const int sensorPin = A0;   // Sharp IR sensor output -> A0
const int servoPin  = 9;    // Servo control pin -> 9
float scdistance = 0.0;
float scpitch = 0.0;
float roll = 0, pitch = 0;

const uint8_t NUM_SAMPLES = 10;
int samples[NUM_SAMPLES];
long runningSum = 0;
uint8_t currentIndex = 0;

Servo myServo;
String inputString = "";
bool stringComplete = false;

unsigned long previousMillis = 0;
const long interval = 50; // 1 millisecond loop for frequent sampling

// ---------- New Variables for Speed Measurement ----------
float lastDistanceCm = 0.0;           // Keep track of the last measured distance (in cm)
unsigned long lastSpeedCalcTime = 0;  // Keep track of last time we computed speed (in ms)

// ---------------------------------------------------------


void setup() {
  Serial.begin(115200);
  while (!Serial) {
    // Wait for Serial to be ready
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

  // Initialize the running average array
  for (uint8_t i = 0; i < NUM_SAMPLES; i++) {
    samples[i] = 0;
    runningSum += samples[i];
  }

  // Initialize our time reference for speed calculation
  lastSpeedCalcTime = millis();
}

// -------------- New Function to Measure Speed --------------
float measureBallSpeed(float currentDistanceCm) {
  /*
    Calculates the speed of the ball in cm/s by looking at
    how much the distance reading has changed since the last
    measurement over the elapsed time.
  */

  // Get current time
  unsigned long currentTime = millis();

  // Compute time difference in seconds
  float deltaTime = (currentTime - lastSpeedCalcTime) / 1000.0; 

  // Compute distance difference
  float deltaDistance = currentDistanceCm - lastDistanceCm; // cm

  // Avoid dividing by zero if the sampling is too fast
  float speed = 0.0;
  if (deltaTime > 0.0) {
    // speed in cm/s
    speed = deltaDistance / deltaTime;  
  }

  // Store the current distance & time for next iteration
  lastDistanceCm = currentDistanceCm;
  lastSpeedCalcTime = currentTime;

  return speed;
}
// -----------------------------------------------------------

void loop() {
  // Read the distance using your running-average function
  float distanceCm = readSharpDistanceCm(sensorPin);

  float xAcc, yAcc, zAcc;

  // Read the serial port for servo commands
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }

  // Convert string from serial to servo angle
  if (stringComplete) {
    int angle = inputString.toInt();  // Convert string to integer
    angle = constrain(angle, 24, 120);
    myServo.write(angle);             // Move servo
    // Clear for next time
    inputString = "";
    stringComplete = false;
  }

  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    // Save the last time you ran the code
    previousMillis = currentMillis;

    // Reading the accelerometer
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(xAcc, yAcc, zAcc);
      pitch = atan2(-xAcc, sqrt(yAcc * yAcc + zAcc * zAcc)) * 180.0 / PI;
    }

    // Map and print distance and pitch to -1 to 1
    scdistance = map(distanceCm * 100, 600, 1500, -1000, 1000);
    scpitch    = map(pitch * 100, -1000, 1100, -1000, 1000);

    // --------------- Compute and Print Speed ---------------
    float ballSpeed = measureBallSpeed(distanceCm); // cm/s
    // Print distance, pitch, and speed
    Serial.print("D");
    Serial.print(scdistance / 1000, 2);
    Serial.print("  P");
    Serial.print(scpitch / 1000, 2);
    Serial.print("  Speed(cm/s): ");
    Serial.println(ballSpeed, 2);
  }
}


float readSharpDistanceCm(int pin) {
  // Remove old sample from the sum
  runningSum -= samples[currentIndex];

  // Get a new raw reading
  int rawValue = analogRead(pin);    // 0 - 1023
  samples[currentIndex] = rawValue;

  // Add new sample to running sum
  runningSum += rawValue;

  // Update circular buffer index
  currentIndex++;
  if (currentIndex >= NUM_SAMPLES) {
    currentIndex = 0;
  }

  // Compute average
  int average = runningSum / NUM_SAMPLES;

  // Convert raw reading to voltage
  float voltage = average * (5.0 / 1023.0);

  // Use the approximate distance formula for Sharp IR
  float distance = 27.728 * pow(voltage, -1.2045);

  // Constrain distance to non-negative
  if (distance < 0) {
    distance = 0;
  }

  return distance;
}
