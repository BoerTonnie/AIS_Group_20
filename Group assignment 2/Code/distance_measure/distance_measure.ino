// Pin Assignments
const int sensorPin = A0;  // Analog input pin to which the sensor is connected

void setup() {
  Serial.begin(9600);       // Initialize serial communication
  pinMode(sensorPin, INPUT);
}

void loop() {
  // Step 1: Read the raw ADC value (0–1023)
  int rawValue = analogRead(sensorPin);

  // Step 2: Convert raw ADC reading to voltage
  float voltage = rawValue * (5.0 / 1023.0);

  // Step 3: Convert voltage to distance (in cm)
  // The GP2Y0A21YK0F has a somewhat inverse exponential relationship
  // A common approximate formula is:
  // distance (cm) = 27.728 * (voltage ^ -1.2045)
  // You can also use custom calibrations or lookup tables.
  
  float distance = 27.728 * pow(voltage, -1.2045); // approximate
    
  // (Optional) clamp the distance so it doesn't show silly values
  if (distance < 0) {
    distance = 0;
  }

  // Step 4: Print results
  Serial.print("Raw Value: ");
  Serial.print(rawValue);
  Serial.print("  |  Voltage: ");
  Serial.print(voltage);
  Serial.print(" V");
  Serial.print("  |  Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  delay(200); // small delay for stability
}
