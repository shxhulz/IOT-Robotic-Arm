#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Create PWM driver instance with default address 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Define constants for servos
#define SERVO_COUNT 9      // Number of servos connected
#define SERVO_FREQ 50      // Standard servo frequency (50Hz)
#define SERVOMIN 100       // This is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX 600       // This is the 'maximum' pulse length count (out of 4096)
#define MOVEMENT_SPEED 100 // Delay between steps in milliseconds (lower = faster)
#define STEP_SIZE 1        // Size of each step in degrees (smaller = smoother, but slower)

// Ultrasonic sensor pins
#define TRIG_PIN 7        // Connect ultrasonic TRIG pin to Arduino digital pin 7 Green Wire
#define ECHO_PIN 8        // Connect ultrasonic ECHO pin to Arduino digital pin 8 white wire

// Ultrasonic sensor variables
unsigned long distanceSum = 0;
int distanceCount = 0;
unsigned long lastDistanceTime = 0;
bool measuringDistance = false;

// Buffer for incoming serial data
const int MAX_BUFFER = 64;
char inputBuffer[MAX_BUFFER];
int bufferIndex = 0;

// Current position of each servo (in degrees)
int servoPositions[SERVO_COUNT] = {90, 90, 90, 90, 90, 90, 70, 90, 90}; // Initialize all servos to 90 degrees

// Target positions for smooth movement
int servoTargets[SERVO_COUNT] = {90, 90, 90, 90, 90, 90, 70, 90, 90};

// Flag to indicate if any servo needs to move
bool movementNeeded = false;

// Flag to track if we're executing a movement sequence
bool executingMovement = false;

// Flag to track if a new command was received
bool newCommandReceived = false;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  Serial.println("Robotic Arm Servo Controller with Smooth Movement and Ultrasonic Sensor");
  Serial.println("Format: 'S<servo_num>A<angle>' (e.g., 'S1A120' sets servo 1 to 120 degrees)");
  Serial.println("Multiple commands can be sent at once (e.g., 'S1A90S2A45S3A180')");
  Serial.println("Send 'GET' to measure distance for one minute and receive average");
  Serial.println("Send 'DIST' to get immediate distance measurement");
  
  // Setup ultrasonic sensor pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  
  // Initialize PWM driver
  pwm.begin();
  
  // Configure oscillator and frequency
  pwm.setOscillatorFrequency(27000000); // 27MHz
  pwm.setPWMFreq(SERVO_FREQ);           // 50Hz for standard servos
  
  // Use faster I2C communication (400kHz)
  Wire.setClock(400000);
  
  // Initialize all servos to their default positions
  for (int i = 0; i < SERVO_COUNT; i++) {
    setServoAngle(i, servoPositions[i]);
    delay(100); // Small delay between servo movements
  }
  
  delay(500); // Give servos time to reach initial positions
  Serial.println("Initialization complete - Ready for commands");
}

void loop() {
  // Check for serial data
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    // If we get a newline or carriage return, process the command
    if (c == '\n' || c == '\r') {
      if (bufferIndex > 0) {
        inputBuffer[bufferIndex] = '\0'; // Null-terminate the string
        
        // Check for distance measurement commands
        if (strcmp(inputBuffer, "GET") == 0) {
          startDistanceMeasurement();
        } 
        else if (strcmp(inputBuffer, "DIST") == 0) {
          sendImmediateDistance();
        }
        else {
          // Parse and set targets for servo movement
          parseAndExecuteCommand(inputBuffer);
          
          // Set flags for new movement
          executingMovement = true;
          newCommandReceived = true;
        }
        
        // Reset buffer
        bufferIndex = 0;
      }
    } 
    // Otherwise, add the character to the buffer if there's space
    else if (bufferIndex < MAX_BUFFER - 1) {
      inputBuffer[bufferIndex++] = c;
    }
  }
  
  // Handle smooth movement of servos
  updateServoPositions();
  
  // Handle distance measurements if active
  if (measuringDistance) {
    updateDistanceMeasurement();
  }
}

// Get and send immediate distance measurement
void sendImmediateDistance() {
  int distance = measureDistance();
  Serial.print("DISTC");
  Serial.println(distance);
}

// Start the distance measurement process
void startDistanceMeasurement() {
  Serial.println("Starting distance measurement for 60 seconds...");
  measuringDistance = true;
  distanceSum = 0;
  distanceCount = 0;
  lastDistanceTime = millis();
}

// Update distance measurements over time
void updateDistanceMeasurement() {
  // Take a measurement every second
  unsigned long currentTime = millis();
  if (currentTime - lastDistanceTime >= 1000) {
    // Get a distance reading
    int distance = measureDistance();
    
    // Add to our running total
    distanceSum += distance;
    distanceCount++;
    
    lastDistanceTime = currentTime;
    
    // Check if we've completed our minute of measurements
    if (distanceCount >= 60) {
      // Calculate average distance
      int avgDistance = distanceSum / distanceCount;
      
      // Send the formatted result
      Serial.print("DISTC");
      Serial.println(avgDistance);
      
      // Reset measurement state
      measuringDistance = false;
    }
  }
}

// Measure distance using the ultrasonic sensor
int measureDistance() {
  // Clear the trigger pin
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  
  // Set the trigger pin HIGH for 10 microseconds
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  // Read the echo pin, returns the sound wave travel time in microseconds
  long duration = pulseIn(ECHO_PIN, HIGH);
  
  // Calculate the distance
  // Speed of sound is approximately 343 meters per second or 0.0343 cm/microsecond
  // Distance = (time × speed) / 2 (divided by 2 because sound travels to object and back)
  int distance = duration * 0.0343 / 2;
  
  return distance;
}

// Update servo positions with smooth movement
void updateServoPositions() {
  // Reset movement flag for this cycle
  movementNeeded = false;
  
  // Check each servo to see if it needs to move toward target
  for (int i = 0; i < SERVO_COUNT; i++) {
    if (servoPositions[i] != servoTargets[i]) {
      movementNeeded = true;
      
      // Determine direction and move one step
      if (servoPositions[i] < servoTargets[i]) {
        servoPositions[i] = min(servoPositions[i] + STEP_SIZE, servoTargets[i]);
      } else {
        servoPositions[i] = max(servoPositions[i] - STEP_SIZE, servoTargets[i]);
      }
      
      // Update servo position
      setServoAngle(i, servoPositions[i]);
    }
  }
  
  // If any servo moved, add a small delay
  if (movementNeeded) {
    delay(MOVEMENT_SPEED);
  }
  // If we were executing a movement and now all servos are at their targets
  else if (executingMovement) {
    // Print confirmation only once per command sequence
    Serial.println("OK");
    
    // Print the final positions for verification
    // for (int i = 0; i < SERVO_COUNT; i++) {
    //   Serial.print("Servo ");
    //   Serial.print(i + 1);
    //   Serial.print(": ");
    //   Serial.print(servoPositions[i]);
    //   Serial.println(" degrees");
    // }
    
    // Reset movement flag
    executingMovement = false;
  }
}

// Parse and execute commands from the input buffer
void parseAndExecuteCommand(char* buffer) {
  int index = 0;
  bool anyValidCommand = false;
  
  Serial.print("Received command: ");
  Serial.println(buffer);
  
  // Continue parsing until we reach the end of the buffer
  while (buffer[index] != '\0') {
    // Look for 'S' followed by servo number
    if (buffer[index] == 'S' || buffer[index] == 's') {
      index++; // Move past 'S'
      
      // Extract servo number
      int servoNum = 0;
      while (isDigit(buffer[index])) {
        servoNum = servoNum * 10 + (buffer[index] - '0');
        index++;
      }
      
      // Adjust for zero-based indexing
      servoNum--; 
      
      // Check if servo number is valid
      if (servoNum >= 0 && servoNum < SERVO_COUNT) {
        // Look for 'A' followed by angle
        if (buffer[index] == 'A' || buffer[index] == 'a') {
          index++; // Move past 'A'
          
          // Extract angle
          int angle = 0;
          while (isDigit(buffer[index])) {
            angle = angle * 10 + (buffer[index] - '0');
            index++;
          }
          
          // Check if angle is valid
          if (angle >= 0 && angle <= 180) {
            // Set target angle (actual movement happens in updateServoPositions)
            servoTargets[servoNum] = angle;
            anyValidCommand = true;
            
            Serial.print("Setting Servo ");
            Serial.print(servoNum + 1);
            Serial.print(" target to ");
            Serial.print(angle);
            Serial.println(" degrees");
          } else {
            Serial.print("Error: Angle must be between 0 and 180 degrees (received ");
            Serial.print(angle);
            Serial.println(")");
          }
        } else {
          // Skip invalid characters
          index++;
        }
      } else {
        Serial.print("Error: Invalid servo number ");
        Serial.print(servoNum + 1);
        Serial.print(". Must be between 1 and ");
        Serial.println(SERVO_COUNT);
        // Skip this command
        while (buffer[index] != '\0' && buffer[index] != 'S' && buffer[index] != 's') {
          index++;
        }
      }
    } else {
      // Skip any other characters
      index++;
    }
  }
  
  if (anyValidCommand) {
    Serial.println("Executing smooth movement...");
  } else {
    Serial.println("No valid commands found in input");
  }
}

// Convert angle (0-180) to pulse length (SERVOMIN-SERVOMAX)
int angleToPulse(int angle) {
  // Constrain angle to valid range
  angle = constrain(angle, 0, 180);
  
  // Map angle from 0-180 to SERVOMIN-SERVOMAX
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

// Set a servo to a specific angle
void setServoAngle(int servoNum, int angle) {
  // Additional check to ensure servo number is valid
  if (servoNum >= 0 && servoNum < SERVO_COUNT) {
    int pulse = angleToPulse(angle);
    pwm.setPWM(servoNum, 0, pulse);
  }
}