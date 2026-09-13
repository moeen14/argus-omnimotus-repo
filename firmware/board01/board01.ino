#include <Wire.h>
#include <SPI.h>
#include <ESP32Servo.h>
#include <math.h>
#include <VL53L1X.h>

#include <Adafruit_LSM6DSOX.h>
#include <Adafruit_LIS3MDL.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>


#define SERIAL_BAUD 921600
#define SERIAL_TIMEOUT_MS 20
#define FW_VERSION "board01_final_v4"

#define I2C_SDA 21
#define I2C_SCL 22
#define I2C_CLOCK_HZ 100000

#define ENC_SCK  18
#define ENC_MISO 19
#define ENC_MOSI 23

#define ENC_CS0  5
#define ENC_CS1  4
#define ENC_CS2  33
#define ENC_CS3  32

#define TOF1_XSHUT 13
#define TOF2_XSHUT 14
#define TOF3_XSHUT 16
#define TOF4_XSHUT 17
#define TOF5_XSHUT 25
#define TOF6_XSHUT 26
#define TOF7_XSHUT 27

#define SG90_SERVO_PIN 15
#define SG90_SERVO_FREQ_HZ 50
#define SG90_SERVO_MIN_US 500
#define SG90_SERVO_MAX_US 2500
#define SG90_INITIAL_ANGLE 180

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDR 0x3C

const int NUM_ENCODERS = 4;
const int NUM_TOF = 7;

const int CS_PINS[NUM_ENCODERS] = {ENC_CS0, ENC_CS1, ENC_CS2, ENC_CS3};
const int TOF_XSHUT_PINS[NUM_TOF] = {
  TOF1_XSHUT,
  TOF2_XSHUT,
  TOF3_XSHUT,
  TOF4_XSHUT,
  TOF5_XSHUT,
  TOF6_XSHUT,
  TOF7_XSHUT
};

const uint8_t TOF_ADDRESSES[NUM_TOF] = {
  0x30,
  0x31,
  0x32,
  0x33,
  0x34,
  0x35,
  0x36
};

const char *TOF_LABELS[NUM_TOF] = {
  "FRS",
  "BRS",
  "BLS",
  "FLF",
  "FLS",
  "CAM",
  "FRF"
};

const unsigned long RPM_INTERVAL_MS = 50;
const unsigned long TOF_INTERVAL_MS = 100;
const unsigned long IMU_INTERVAL_MS = 20;

Servo sg90Servo;
int sg90CurrentAngle = SG90_INITIAL_ANGLE;

const float MAX_REASONABLE_RPM = 500.0f;
const float RPM_DEADBAND = 1.0f;
const float RPM_SMOOTH_ALPHA = 0.35f;
const float MAX_SINGLE_SAMPLE_JUMP = 250.0f;
const uint16_t TOF_VALID_MIN_MM = 20;
const uint16_t TOF_VALID_MAX_MM = 2500;
const uint16_t TOF_MAX_STEP_PER_READ_MM = 80;
const int TOF_MEDIAN_WINDOW = 5;
const uint16_t TOF_INIT_TIMEOUT_MS = 200;
const uint16_t TOF_RUNTIME_TIMEOUT_MS = 60;
const uint8_t TOF_MAX_CONSECUTIVE_TIMEOUTS = 5;

Adafruit_LSM6DSOX lsm6dsox;
Adafruit_LIS3MDL lis3mdl;
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
VL53L1X tof[NUM_TOF];

bool lsmOK = false;
bool magOK = false;
bool oledOK = false;
bool tofOK[NUM_TOF] = {false, false, false, false, false, false, false};
bool tofTimeout[NUM_TOF] = {false, false, false, false, false, false, false};
uint8_t tofConsecutiveTimeouts[NUM_TOF] = {0, 0, 0, 0, 0, 0, 0};

unsigned long lastRpmMs = 0;
unsigned long lastTofMs = 0;
unsigned long lastImuMs = 0;

float lastAngle[NUM_ENCODERS] = {0, 0, 0, 0};
float wheelRpm[NUM_ENCODERS] = {0, 0, 0, 0};
uint16_t tofRawMm[NUM_TOF] = {0, 0, 0, 0, 0, 0, 0};
uint16_t tofMm[NUM_TOF] = {0, 0, 0, 0, 0, 0, 0};
uint16_t tofSamplesMm[NUM_TOF][TOF_MEDIAN_WINDOW] = {
  {0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0}
};
uint8_t tofSampleIndex[NUM_TOF] = {0, 0, 0, 0, 0, 0, 0};
uint8_t tofSampleCount[NUM_TOF] = {0, 0, 0, 0, 0, 0, 0};
bool tofStableValid[NUM_TOF] = {false, false, false, false, false, false, false};

unsigned long greenCount = 0;
unsigned long blueCount = 0;
unsigned long redCount = 0;
unsigned long totalCount = 0;

enum OledScreen {
  OLED_SCREEN_TEAM,
  OLED_SCREEN_COUNTERS
};

OledScreen currentOledScreen = OLED_SCREEN_TEAM;

void setupOLED();
void setupOLED(bool playAnimation);
void restoreCurrentOledScreen();

int countOkTofSensors() {
  int ok = 0;
  for (int i = 0; i < NUM_TOF; i++) {
    if (tofOK[i]) ok++;
  }
  return ok;
}

void printBoardStatus(const char *prefix) {
  Serial.print(prefix);
  Serial.print(",BOARD01_FINAL,");
  Serial.print(FW_VERSION);
  Serial.print(",tof=");
  Serial.print(countOkTofSensors());
  Serial.print("/");
  Serial.print(NUM_TOF);
  Serial.print(",lsm=");
  Serial.print(lsmOK ? "1" : "0");
  Serial.print(",mag=");
  Serial.print(magOK ? "1" : "0");
  Serial.print(",oled=");
  Serial.print(oledOK ? "1" : "0");
  Serial.print(",enc=");
  Serial.print(NUM_ENCODERS);
  Serial.print(",sg90=");
  Serial.println(sg90CurrentAngle);
}


uint16_t readAS5048RawNoTransaction(int csPin) {
  digitalWrite(csPin, LOW);
  delayMicroseconds(2);
  uint16_t value = SPI.transfer16(0xFFFF);
  digitalWrite(csPin, HIGH);
  return value & 0x3FFF;
}

uint16_t readAS5048Raw(int csPin) {
  SPI.beginTransaction(SPISettings(4000000, MSBFIRST, SPI_MODE1));
  uint16_t value = readAS5048RawNoTransaction(csPin);
  SPI.endTransaction();
  return value;
}


void readAllEncodersRawDeg(float outDeg[NUM_ENCODERS]) {
  SPI.beginTransaction(SPISettings(4000000, MSBFIRST, SPI_MODE1));
  for (int i = 0; i < NUM_ENCODERS; i++) {
    outDeg[i] = rawToDeg(readAS5048RawNoTransaction(CS_PINS[i]));
  }
  SPI.endTransaction();
}

float rawToDeg(uint16_t raw) {
  return (raw * 360.0f) / 16383.0f;
}

float angleDelta(float prev, float curr) {
  float d = curr - prev;
  while (d > 180.0f) d -= 360.0f;
  while (d < -180.0f) d += 360.0f;
  return d;
}

bool validTofDistance(uint16_t mm) {
  return mm >= TOF_VALID_MIN_MM && mm <= TOF_VALID_MAX_MM;
}

uint16_t medianTofForSensor(int sensor) {
  uint16_t sorted[TOF_MEDIAN_WINDOW];
  uint8_t count = tofSampleCount[sensor];

  for (uint8_t i = 0; i < count; i++) {
    sorted[i] = tofSamplesMm[sensor][i];
  }

  for (uint8_t i = 1; i < count; i++) {
    uint16_t value = sorted[i];
    int j = i - 1;
    while (j >= 0 && sorted[j] > value) {
      sorted[j + 1] = sorted[j];
      j--;
    }
    sorted[j + 1] = value;
  }

  return sorted[count / 2];
}

uint16_t limitTofStep(uint16_t current, uint16_t target) {
  if (target > current) {
    uint16_t delta = target - current;
    return current + min(delta, TOF_MAX_STEP_PER_READ_MM);
  }

  uint16_t delta = current - target;
  return current - min(delta, TOF_MAX_STEP_PER_READ_MM);
}

void disableTofSensor(int sensor, const char *reason) {
  if (sensor < 0 || sensor >= NUM_TOF) return;

  tofOK[sensor] = false;
  tofTimeout[sensor] = false;
  tofRawMm[sensor] = 0;
  tofMm[sensor] = 0;
  tofSampleIndex[sensor] = 0;
  tofSampleCount[sensor] = 0;
  tofStableValid[sensor] = false;
  tofConsecutiveTimeouts[sensor] = 0;

  pinMode(TOF_XSHUT_PINS[sensor], OUTPUT);
  digitalWrite(TOF_XSHUT_PINS[sensor], LOW);

  Serial.print("WARN,TOF_DISABLED,");
  Serial.print(sensor);
  Serial.print(",");
  Serial.print(TOF_LABELS[sensor]);
  Serial.print(",");
  Serial.println(reason);
}

void setupEncoders() {
  SPI.begin(ENC_SCK, ENC_MISO, ENC_MOSI);

  for (int i = 0; i < NUM_ENCODERS; i++) {
    pinMode(CS_PINS[i], OUTPUT);
    digitalWrite(CS_PINS[i], HIGH);
  }

  for (int i = 0; i < NUM_ENCODERS; i++) {
    lastAngle[i] = rawToDeg(readAS5048Raw(CS_PINS[i]));
  }

  lastRpmMs = millis();
}

void setupIMU() {
  lsmOK = false;
  magOK = false;

  lsmOK = lsm6dsox.begin_I2C();
  if (lsmOK) {
    lsm6dsox.setAccelRange(LSM6DS_ACCEL_RANGE_4_G);
    lsm6dsox.setGyroRange(LSM6DS_GYRO_RANGE_500_DPS);
    lsm6dsox.setAccelDataRate(LSM6DS_RATE_104_HZ);
    lsm6dsox.setGyroDataRate(LSM6DS_RATE_104_HZ);
  }

  magOK = lis3mdl.begin_I2C();
  if (magOK) {
    lis3mdl.setPerformanceMode(LIS3MDL_MEDIUMMODE);
    lis3mdl.setOperationMode(LIS3MDL_CONTINUOUSMODE);
    lis3mdl.setDataRate(LIS3MDL_DATARATE_155_HZ);
    lis3mdl.setRange(LIS3MDL_RANGE_4_GAUSS);
  }
}

void setupTofSensors() {
  for (int i = 0; i < NUM_TOF; i++) {
    pinMode(TOF_XSHUT_PINS[i], OUTPUT);
    digitalWrite(TOF_XSHUT_PINS[i], LOW);
    tof[i] = VL53L1X();
    tofOK[i] = false;
    tofRawMm[i] = 0;
    tofMm[i] = 0;
    tofSampleIndex[i] = 0;
    tofSampleCount[i] = 0;
    tofStableValid[i] = false;
    tofTimeout[i] = false;
    tofConsecutiveTimeouts[i] = 0;
  }

  delay(500);

  for (int i = 0; i < NUM_TOF; i++) {
    digitalWrite(TOF_XSHUT_PINS[i], HIGH);
    delay(250);

    tof[i].setTimeout(TOF_INIT_TIMEOUT_MS);

    if (!tof[i].init()) {
      Serial.print("WARN,TOF_INIT_FAIL,");
      Serial.print(i);
      Serial.print(",");
      Serial.println(TOF_LABELS[i]);
      disableTofSensor(i, "init_fail");
      continue;
    }

    tof[i].setAddress(TOF_ADDRESSES[i]);
    delay(80);
    tof[i].setDistanceMode(VL53L1X::Short);
    tof[i].setMeasurementTimingBudget(100000);
    tof[i].startContinuous(100);
    tof[i].setTimeout(TOF_RUNTIME_TIMEOUT_MS);
    tofOK[i] = true;
  }
}

void resetAllTofSensors() {
  Serial.println("ACK,TOF_RESET,START");

  for (int i = 0; i < NUM_TOF; i++) {
    pinMode(TOF_XSHUT_PINS[i], OUTPUT);
    digitalWrite(TOF_XSHUT_PINS[i], LOW);
  }

  Wire.end();
  delay(50);
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(I2C_CLOCK_HZ);

  setupIMU();
  setupOLED(false);
  setupTofSensors();
  lastTofMs = millis();
  lastImuMs = millis();

  Serial.print("ACK,TOF_RESET,DONE,");
  Serial.print(countOkTofSensors());
  Serial.print("/");
  Serial.println(NUM_TOF);
}

void drawCenteredText(const char *text, int y, int size) {
  int16_t x1, y1;
  uint16_t w, h;
  display.setTextSize(size);
  display.getTextBounds(text, 0, y, &x1, &y1, &w, &h);
  int x = (SCREEN_WIDTH - (int)w) / 2;
  display.setCursor(x, y);
  display.print(text);
}

void drawBoldText(int x, int y, const char *text, int size) {
  display.setTextSize(size);
  display.setCursor(x, y);
  display.print(text);
  display.setCursor(x + 1, y);
  display.print(text);
}

void drawCenteredBoldText(const char *text, int y, int size) {
  int16_t x1, y1;
  uint16_t w, h;
  display.setTextSize(size);
  display.getTextBounds(text, 0, y, &x1, &y1, &w, &h);
  int x = (SCREEN_WIDTH - (int)w) / 2;
  drawBoldText(x, y, text, size);
}

void drawEyeFrame(int openHeight) {
  display.clearDisplay();
  display.drawRoundRect(15, 22, 98, openHeight, openHeight / 2, SSD1306_WHITE);
  display.fillCircle(64, 22 + openHeight / 2, max(2, openHeight / 4), SSD1306_WHITE);
  display.display();
}

void drawTinyRobot(int x, int y) {
  display.drawRect(x, y, 9, 6, SSD1306_WHITE);
  display.drawPixel(x + 2, y + 2, SSD1306_WHITE);
  display.drawPixel(x + 6, y + 2, SSD1306_WHITE);
  display.drawFastHLine(x + 2, y + 7, 5, SSD1306_WHITE);
}

void playASABESplash() {
  const unsigned long durationMs = 10000;
  unsigned long startMs = millis();
  int frame = 0;

  while (millis() - startMs < durationMs) {
    int drift = frame % (SCREEN_WIDTH + 24);
    display.clearDisplay();

    for (int i = 0; i < 5; i++) {
      int x = (drift + i * 29) % (SCREEN_WIDTH + 18) - 9;
      int y = 5 + ((i * 13 + frame / 2) % 52);
      int r = 1 + ((frame + i) % 3);
      display.drawCircle(x, y, r, SSD1306_WHITE);
    }

    drawTinyRobot((SCREEN_WIDTH - (drift % (SCREEN_WIDTH + 18))) - 9, 50);
    drawTinyRobot((drift + 35) % (SCREEN_WIDTH + 18) - 9, 7);

    display.fillRect(16, 0, 96, 64, SSD1306_BLACK);
    drawCenteredText("ASABE", 2, 2);
    drawCenteredText("Student", 22, 1);
    drawCenteredText("Robotics", 34, 1);
    drawCenteredText("Challenge", 46, 1);
    display.display();

    frame++;
    delay(80);
  }
}

void drawStaticNoise(int speckles, int bands) {
  display.clearDisplay();

  for (int i = 0; i < speckles; i++) {
    int x = random(0, SCREEN_WIDTH);
    int y = random(0, SCREEN_HEIGHT);
    display.drawPixel(x, y, SSD1306_WHITE);
  }

  for (int i = 0; i < bands; i++) {
    int y = random(0, SCREEN_HEIGHT);
    int x = random(-20, 70);
    int w = random(25, 90);
    display.drawFastHLine(max(0, x), y, min(w, SCREEN_WIDTH), SSD1306_WHITE);
  }
}

void drawGlitchedTeamName(int offset) {
  drawStaticNoise(90, 4);
  drawCenteredText("THE", 0 + offset, 1);
  drawCenteredBoldText("MAROON", 14 + offset, 2);
  drawCenteredText("Robotics Team", 43 - offset, 1);
  display.display();
}

void showTeamName() {
  if (!oledOK) return;
  display.clearDisplay();
  drawCenteredText("THE", 0, 1);
  drawCenteredBoldText("MAROON", 14, 2);
  display.drawLine(15, 36, 112, 36, SSD1306_WHITE);
  drawCenteredText("Robotics Team", 45, 1);
  display.display();
  currentOledScreen = OLED_SCREEN_TEAM;
}

void playStaticReveal() {
  for (int i = 0; i < 12; i++) {
    drawStaticNoise(160, 8);
    display.display();
    delay(45);
  }

  for (int i = 0; i < 8; i++) {
    int offset = (i % 2 == 0) ? -2 : 2;
    drawGlitchedTeamName(offset);
    delay(80);
  }

  for (int i = 0; i < 3; i++) {
    showTeamName();
    delay(160);
    drawStaticNoise(120, 5);
    display.display();
    delay(70);
  }
}

void playScreenTransition() {
  if (!oledOK) return;
  for (int i = 0; i < 5; i++) {
    drawStaticNoise(120, 6);
    display.display();
    delay(35);
  }

  for (int i = 0; i < 3; i++) {
    display.clearDisplay();
    display.drawFastHLine(0, 12 + i * 15, SCREEN_WIDTH, SSD1306_WHITE);
    display.drawFastHLine(random(0, 30), random(0, SCREEN_HEIGHT), random(40, 128), SSD1306_WHITE);
    display.display();
    delay(45);
  }
}

void playBootAnimation() {
  if (!oledOK) return;
  for (int h = 2; h <= 24; h += 4) {
    drawEyeFrame(h);
    delay(90);
  }

  delay(180);
  playASABESplash();
  playStaticReveal();
  showTeamName();
}

void showCounters() {
  if (!oledOK) return;
  display.clearDisplay();
  drawBoldText(0, 0, "COLOR COUNT", 1);
  display.drawLine(0, 12, 127, 12, SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 18);
  display.print("Green: ");
  display.print(greenCount);

  display.setCursor(0, 30);
  display.print("Blue : ");
  display.print(blueCount);

  display.setCursor(0, 42);
  display.print("Red  : ");
  display.print(redCount);

  display.setCursor(0, 54);
  display.print("Total: ");
  display.print(totalCount);
  display.display();
  currentOledScreen = OLED_SCREEN_COUNTERS;
}

void resetCounters() {
  greenCount = 0;
  blueCount = 0;
  redCount = 0;
  totalCount = 0;
}

void flashColorWord(const char *word) {
  if (!oledOK) return;
  display.clearDisplay();
  drawCenteredText(word, 20, 2);
  display.display();
  delay(200);
  display.clearDisplay();
  display.display();
  showCounters();
}

void handleColor(char color) {
  totalCount++;
  if (color == 'G') {
    greenCount++;
    flashColorWord("GREEN");
  } else if (color == 'B') {
    blueCount++;
    flashColorWord("BLUE");
  } else if (color == 'R') {
    redCount++;
    flashColorWord("RED");
  }

  Serial.print("ACK,COLOR,");
  Serial.print(color);
  Serial.print(",");
  Serial.print(greenCount);
  Serial.print(",");
  Serial.print(blueCount);
  Serial.print(",");
  Serial.print(redCount);
  Serial.print(",");
  Serial.println(totalCount);
}

void drawConfettiFrame(bool showText) {
  display.clearDisplay();

  for (int i = 0; i < 22; i++) {
    int x = random(0, SCREEN_WIDTH);
    int y = random(0, SCREEN_HEIGHT);
    int piece = random(0, 3);
    if (piece == 0) {
      display.drawPixel(x, y, SSD1306_WHITE);
    } else if (piece == 1) {
      display.fillRect(x, y, 2, 2, SSD1306_WHITE);
    } else {
      display.drawFastHLine(x, y, 3, SSD1306_WHITE);
    }
  }

  if (showText) {
    drawCenteredBoldText("FINISH!", 26, 2);
  }

  display.display();
}

void playFinishCelebration() {
  if (!oledOK) return;
  const unsigned long durationMs = 10000;
  unsigned long startMs = millis();
  int frame = 0;

  while (millis() - startMs < durationMs) {
    bool showText = (frame % 6) < 4;
    drawConfettiFrame(showText);
    frame++;
    delay(80);
  }

  restoreCurrentOledScreen();
}

void setupOLED() {
  setupOLED(true);
}

void setupOLED(bool playAnimation) {
  oledOK = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  if (!oledOK) {
    Serial.println("WARN,OLED_INIT_FAIL");
    return;
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  if (playAnimation) {
    playBootAnimation();
  } else {
    restoreCurrentOledScreen();
  }
}

void restoreCurrentOledScreen() {
  if (currentOledScreen == OLED_SCREEN_COUNTERS) {
    showCounters();
  } else {
    showTeamName();
  }
}

void updateAndPrintRpm() {
  unsigned long now = millis();
  if (now - lastRpmMs < RPM_INTERVAL_MS) return;

  float dt = (now - lastRpmMs) / 1000.0f;
  lastRpmMs = now;

  float freshAngle[NUM_ENCODERS];
  readAllEncodersRawDeg(freshAngle);

  for (int i = 0; i < NUM_ENCODERS; i++) {
    float angle = freshAngle[i];
    float delta = angleDelta(lastAngle[i], angle);
    float candidate = fabs((delta / 360.0f) * (60.0f / dt));

    if (candidate <= MAX_REASONABLE_RPM) {
      lastAngle[i] = angle;

      if (candidate < RPM_DEADBAND) {
        wheelRpm[i] = 0.0f;
      } else if (wheelRpm[i] <= RPM_DEADBAND ||
                 fabs(candidate - wheelRpm[i]) <= MAX_SINGLE_SAMPLE_JUMP) {
        wheelRpm[i] = (wheelRpm[i] * (1.0f - RPM_SMOOTH_ALPHA)) + (candidate * RPM_SMOOTH_ALPHA);
      }
    }
  }

  Serial.print("RPM [");
  Serial.print("BR:");
  Serial.print((int)round(wheelRpm[0]));
  Serial.print("  FR:");
  Serial.print((int)round(wheelRpm[1]));
  Serial.print("  BL:");
  Serial.print((int)round(wheelRpm[2]));
  Serial.print("  FL:");
  Serial.print((int)round(wheelRpm[3]));
  Serial.println("]");
}

void readTofSensors() {
  for (int i = 0; i < NUM_TOF; i++) {
    if (!tofOK[i]) {
      tofRawMm[i] = 0;
      tofMm[i] = 0;
      tofStableValid[i] = false;
      tofTimeout[i] = false;
      tofConsecutiveTimeouts[i] = 0;
      continue;
    }

    tofRawMm[i] = tof[i].read(false);
    tofTimeout[i] = tof[i].timeoutOccurred();

    if (tofTimeout[i]) {
      if (tofConsecutiveTimeouts[i] < TOF_MAX_CONSECUTIVE_TIMEOUTS) {
        tofConsecutiveTimeouts[i]++;
      }

      if (tofConsecutiveTimeouts[i] >= TOF_MAX_CONSECUTIVE_TIMEOUTS) {
        disableTofSensor(i, "runtime_timeout");
      }
      continue;
    }

    if (!validTofDistance(tofRawMm[i])) {
      continue;
    }

    tofConsecutiveTimeouts[i] = 0;

    tofSamplesMm[i][tofSampleIndex[i]] = tofRawMm[i];
    tofSampleIndex[i] = (tofSampleIndex[i] + 1) % TOF_MEDIAN_WINDOW;
    if (tofSampleCount[i] < TOF_MEDIAN_WINDOW) {
      tofSampleCount[i]++;
    }

    uint16_t medianMm = medianTofForSensor(i);
    if (!tofStableValid[i]) {
      tofMm[i] = medianMm;
      tofStableValid[i] = true;
    } else {
      tofMm[i] = limitTofStep(tofMm[i], medianMm);
    }
  }
}

void printTof() {
  unsigned long now = millis();
  if (now - lastTofMs < TOF_INTERVAL_MS) return;
  lastTofMs = now;

  readTofSensors();

  Serial.print("TOF,[");
  for (int i = 0; i < NUM_TOF; i++) {
    Serial.print(TOF_LABELS[i]);
    Serial.print(":");

    if (!tofOK[i]) {
      Serial.print("INIT_FAIL");
    } else if (!tofStableValid[i]) {
      if (tofTimeout[i]) {
        Serial.print("TIMEOUT");
      } else {
        Serial.print("NO_VALID");
      }
    } else {
      Serial.print(tofMm[i]);
    }

    if (i < NUM_TOF - 1) {
      Serial.print(",");
    }
  }
  Serial.println("]");
}

void printRawIMU() {
  unsigned long now = millis();
  if (now - lastImuMs < IMU_INTERVAL_MS) return;
  lastImuMs = now;

  Serial.print("IMU");

  if (!lsmOK || !magOK) {
    Serial.println(",ERR");
    return;
  }

  sensors_event_t accel;
  sensors_event_t gyro;
  sensors_event_t temp;
  sensors_event_t mag;

  lsm6dsox.getEvent(&accel, &gyro, &temp);
  lis3mdl.getEvent(&mag);

  Serial.print(",");
  Serial.print(accel.acceleration.x, 3);
  Serial.print(",");
  Serial.print(accel.acceleration.y, 3);
  Serial.print(",");
  Serial.print(accel.acceleration.z, 3);
  Serial.print(",");
  Serial.print(gyro.gyro.x * 57.2957795f, 3);
  Serial.print(",");
  Serial.print(gyro.gyro.y * 57.2957795f, 3);
  Serial.print(",");
  Serial.print(gyro.gyro.z * 57.2957795f, 3);
  Serial.print(",");
  Serial.print(mag.magnetic.x, 3);
  Serial.print(",");
  Serial.print(mag.magnetic.y, 3);
  Serial.print(",");
  Serial.println(mag.magnetic.z, 3);
}

void setupSg90Servo() {
  sg90Servo.setPeriodHertz(SG90_SERVO_FREQ_HZ);
  sg90Servo.attach(SG90_SERVO_PIN, SG90_SERVO_MIN_US, SG90_SERVO_MAX_US);
  sg90Servo.write(sg90CurrentAngle);
  delay(300);
}

void moveSg90ServoTo(int angle) {
  sg90CurrentAngle = constrain(angle, 0, 180);
  sg90Servo.write(sg90CurrentAngle);
  Serial.print("ACK,SG90,");
  Serial.println(sg90CurrentAngle);
}

bool parseSg90Command(const String &input) {
  if (!input.startsWith("SG90")) return false;

  String angleText = input.substring(4);
  angleText.trim();
  if (angleText.startsWith(",") || angleText.startsWith(":") || angleText.startsWith("=")) {
    angleText = angleText.substring(1);
    angleText.trim();
  }

  if (angleText.length() == 0) {
    Serial.println("ERR,SG90_ANGLE");
    return true;
  }

  for (int i = 0; i < angleText.length(); i++) {
    if (!isDigit(angleText.charAt(i))) {
      Serial.println("ERR,SG90_ANGLE");
      return true;
    }
  }

  int angle = angleText.toInt();
  if (angle < 0 || angle > 180) {
    Serial.println("ERR,SG90_RANGE");
    return true;
  }

  moveSg90ServoTo(angle);
  return true;
}

void handleSerialCommand() {
  if (!Serial.available()) return;

  String input = Serial.readStringUntil('\n');
  input.trim();
  input.toUpperCase();
  if (input.length() == 0) return;

  char cmd = input.charAt(0);
  if (input == "PING" || input == "?") {
    printBoardStatus("PONG");
  } else if (parseSg90Command(input)) {
    return;
  } else if (input == "STATUS") {
    printBoardStatus("STATUS");
  } else if (input == "T0") {
    playScreenTransition();
    resetCounters();
    showCounters();
    Serial.println("ACK,OLED,RESET");
  } else if (cmd == 'I') {
    playScreenTransition();
    showTeamName();
    Serial.println("ACK,OLED,TEAM");
  } else if (cmd == 'T') {
    playScreenTransition();
    showCounters();
    Serial.println("ACK,OLED,COUNTERS");
  } else if (cmd == 'F') {
    resetAllTofSensors();
  } else if (cmd == 'G' || cmd == 'B' || cmd == 'R') {
    handleColor(cmd);
  } else if (input == "Z") {
    playFinishCelebration();
    Serial.println("ACK,OLED,FINISH");
  } else {
    Serial.println("ERR,CMD");
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.setTimeout(SERIAL_TIMEOUT_MS);
  randomSeed(analogRead(34) + micros());

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(I2C_CLOCK_HZ);

  setupOLED();
  setupEncoders();
  setupIMU();
  setupTofSensors();
  setupSg90Servo();

  lastRpmMs = millis();
  lastTofMs = millis();
  lastImuMs = millis();
  Serial.print("READY,BOARD01_FINAL,");
  Serial.println(FW_VERSION);
  printBoardStatus("STATUS");
}

void loop() {
  handleSerialCommand();
  updateAndPrintRpm();
  printTof();
  printRawIMU();
}
