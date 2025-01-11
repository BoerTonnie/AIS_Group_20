// Number of samples for the running average
const uint8_t NUM_SAMPLES = 10;

// Array to store the last 10 samples
int samples[NUM_SAMPLES];

// Sum of the last 10 samples
long runningSum = 0;

// Index that points to where the next reading will be placed
uint8_t currentIndex = 0;

void setup() {
  // Initialize Serial for output
  Serial.begin(9600);

  // Initialize the samples array and runningSum
  for (uint8_t i = 0; i < NUM_SAMPLES; i++) {
    samples[i] = 0;
    runningSum += samples[i];
  }
}

void loop() {
  // 1. Subtract the value that we're going to overwrite from runningSum
  runningSum -= samples[currentIndex];

  // 2. Read a new value from analog pin A0
  int newReading = analogRead(A0);

  // 3. Store the new reading in the array
  samples[currentIndex] = newReading;

  // 4. Add the new reading to runningSum
  runningSum += newReading;

  // 5. Move to the next index in the circular buffer
  currentIndex++;
  if (currentIndex >= NUM_SAMPLES) {
    currentIndex = 0;
  }

  // 6. Calculate the running average of the last 10 samples
  int average = runningSum / NUM_SAMPLES;

  // Print the average for debugging
  Serial.print("Running Average (last 10 samples): ");
  Serial.println(average);

  // Optional delay for readability
  delay(100);
}
