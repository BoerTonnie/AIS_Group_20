// Number of samples for the running average
const uint8_t NUM_SAMPLES = 10;

// Array to store the last 10 samples
int samples[NUM_SAMPLES];

// Sum of the last 10 samples
long runningSum = 0;

// Index that points to where the next reading will be placed


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
  int newReading = analogRead(A0);
  samples[currentIndex] = newReading;
  runningSum += newReading;
  currentIndex++;
  if (currentIndex >= NUM_SAMPLES) {
    currentIndex = 0;
  }
  int average = runningSum / NUM_SAMPLES;

  
  Serial.println(average);

  // Optional delay for readability
  delay(100);
}
