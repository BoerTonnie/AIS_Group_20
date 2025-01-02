#include <Servo.h>

Servo myServo; // Create a Servo object
const int trigPin = 3; // Trig pin of the RCWL-1601
const int echoPin = 4; // Echo pin of the RCWL-1601

void setup() {
  // Initialize the servo
  myServo.attach(2); // Attach the servo to pin 9
  myServo.write(80); // Set the servo to a default position (90 degrees)

  // Initialize ultrasonic sensor pins
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  // Initialize Serial Monitor
  Serial.begin(9600);
}

void loop() {
  // Send a trigger pulse
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Measure the time taken for the echo
  long duration = pulseIn(echoPin, HIGH);

  // Calculate distance (in centimeters)
  float distance = (duration / 2.0) * 0.0343;

  // Output distance to the Serial Monitor
  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // Add a short delay between readings
  delay(500);
}
