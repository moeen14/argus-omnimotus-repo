#include <SCServo.h>
#include <math.h>


#define SERIAL_BAUD 921600
#define SERIAL_TIMEOUT_MS 20
#define RXD2 16
#define TXD2 17
#define FW_VERSION "board02_final_v10"

#define L298N1_IN1 32
#define L298N1_IN2 33
#define L298N1_IN3 13
#define L298N1_IN4 14
#define L298N1_ENA 25
#define L298N1_ENB 26

#define L298N2_IN1 4
#define L298N2_IN2 5
#define L298N2_IN3 18
#define L298N2_IN4 19
#define L298N2_ENA 21
#define L298N2_ENB 22

#define LED_RED   23
#define LED_GREEN 15
#define LED_BLUE  2

#define DISPENSER_SERVO_PIN 27
#define DISPENSER_LDR_PIN   35

SMS_STS st;

const int NUM_SERVOS = 9;
const int IDS[NUM_SERVOS] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
const int NUM_MOTORS = 4;
const float SAFE_TOLERANCE_DEG = 5.0f;
const int NUM_SWERVE = 4;
const float MAX_DELTA_DEG = 30.0f;

enum SwerveMode {
  SWERVE_UNKNOWN,
  SWERVE_INITIAL,
  SWERVE_MID,
  SWERVE_PERPENDICULAR
};

SwerveMode currentSwerveMode = SWERVE_UNKNOWN;


const int WHEEL_TO_SWERVE_IDX[NUM_SWERVE] = {0, 1, 3, 2};
float currentSwerveTargets[NUM_SWERVE] = {37.0f, 6.0f, 145.0f, 221.0f};

const char *swerveModeName() {
  if (currentSwerveMode == SWERVE_INITIAL) return "initial";
  if (currentSwerveMode == SWERVE_MID) return "mid";
  if (currentSwerveMode == SWERVE_PERPENDICULAR) return "perpendicular";
  return "unknown";
}


const int DELTA_FORWARD_SIGN[NUM_SWERVE]  = {-1,  -1, -1,  -1};
const int DELTA_BACKWARD_SIGN[NUM_SWERVE] = { 1, 1,  1, 1};
const int DELTA_RIGHT_SIGN[NUM_SWERVE]    = { 1, 1,  1, 1};
const int DELTA_LEFT_SIGN[NUM_SWERVE]     = {-1,  -1, -1,  -1};

const int PWM_FREQ = 1000;
const int PWM_RESOLUTION = 8;
const int MAX_MOTOR_PWM = 255;


const int MOTOR_FORWARD_SIGN[NUM_MOTORS]  = { -1,  1,  1,  1};
const int MOTOR_BACKWARD_SIGN[NUM_MOTORS] = {1, -1, -1, -1};
const int MOTOR_LEFT_SIGN[NUM_MOTORS]     = {1, 1, 1, -1};
const int MOTOR_RIGHT_SIGN[NUM_MOTORS]    = { -1,  -1,  -1,  1};
const int MOTOR_CW_SIGN[NUM_MOTORS]       = {1, -1,  1,  1};
const int MOTOR_CCW_SIGN[NUM_MOTORS]      = { -1,  1, -1, -1};

struct MotorConfig {
  int in1;
  int in2;
  int en;
};

const MotorConfig MOTORS[NUM_MOTORS] = {
  {L298N1_IN1, L298N1_IN2, L298N1_ENA},
  {L298N1_IN3, L298N1_IN4, L298N1_ENB},
  {L298N2_IN1, L298N2_IN2, L298N2_ENA},
  {L298N2_IN3, L298N2_IN4, L298N2_ENB},
};

struct SafeArcConfig {
  const char *name;
  bool safeArcEnabled;
  float startDeg;
  float endDeg;
  int dir;
  float initialDeg;
  float midDeg;
  float perpendicularDeg;
  float leftLookDeg;
  float rightLookDeg;
};

const SafeArcConfig SAFE_ARCS[NUM_SERVOS] = {
  {"BR",             true,  67.0f, 186.0f, -1,  37.0f, 306.0f, 225.0f, -1.0f,  -1.0f},
  {"FR",             true, 335.0f, 216.0f,  1,   6.0f,  96.0f, 180.0f, -1.0f,  -1.0f},
  {"FL",             true, 251.0f,  11.0f, -1, 221.0f, 131.0f,  41.0f, -1.0f,  -1.0f},
  {"BL",             true, 115.0f, 355.0f,  1, 145.0f, 235.0f, 325.0f, -1.0f,  -1.0f},
  {"ArmHorizontal",  true, 295.0f, 345.0f,  1, 321.0f,  -1.0f,  -1.0f, -1.0f,  -1.0f},
  {"ArmVertical",    true,  60.0f, 240.0f, -1,  35.0f,  -1.0f,  -1.0f, -1.0f,  -1.0f},
  {"Camera",         true, 321.0f, 231.0f,  1,  51.0f,  -1.0f,  -1.0f,141.0f, 321.0f},
  {"Servo8",         true, 167.0f,   5.0f,  1, 176.0f,  -1.0f,  -1.0f, -1.0f,  -1.0f},
  {"Servo9",         true,  23.0f, 148.0f,  1,  20.0f,  -1.0f,  -1.0f, -1.0f,  -1.0f},
};

const unsigned long LOOP_DELAY_MS = 10;
const unsigned long LOG_PERIOD_MS = 100;
const unsigned long MOVE_TIMEOUT_MS = 4000;
const bool SERVO_MOVE_LOGS = false;

struct MotionState {
  bool active;
  float requestedTarget;
  float target;
  float startAngle;
  float startErr;
  float lastErr;
  float maxAbsErr;
  float threshold;
  int minSpeed;
  int maxSpeed;
  int acc;
  int lastSpeed;
  unsigned long startMs;
  unsigned long lastLogMs;
  int samples;
  int stableCount;
};

MotionState motion[NUM_SERVOS];

void configureProfile(MotionState &m);
int computeSpeed(const MotionState &m, float err);
float angleError(float target, float current);
float swerveBaseTarget(int idx, SwerveMode mode);
void commandSwerveMode(SwerveMode mode, const char *label);


float normalizeAngle(float angle) {
  angle = fmodf(angle, 360.0f);
  if (angle < 0.0f) angle += 360.0f;
  return angle;
}

float posToAngle(int pos) {
  pos = pos % 4096;
  if (pos < 0) pos += 4096;
  return pos * 360.0f / 4096.0f;
}

float arcDistance(float from, float to, int dir) {
  from = normalizeAngle(from);
  to = normalizeAngle(to);
  if (dir > 0) {
    return fmodf((to - from + 360.0f), 360.0f);
  }
  return fmodf((from - to + 360.0f), 360.0f);
}

float safeArcLength(int idx) {
  if (!SAFE_ARCS[idx].safeArcEnabled) return 360.0f;
  const SafeArcConfig &cfg = SAFE_ARCS[idx];
  return arcDistance(cfg.startDeg, cfg.endDeg, cfg.dir);
}

float safeArcCoordRaw(int idx, float angle) {
  const SafeArcConfig &cfg = SAFE_ARCS[idx];
  return arcDistance(cfg.startDeg, angle, cfg.dir);
}

bool isInSafeArc(int idx, float angle) {
  if (!SAFE_ARCS[idx].safeArcEnabled) return true;
  float coord = safeArcCoordRaw(idx, angle);
  float len = safeArcLength(idx);
  return coord <= (len + SAFE_TOLERANCE_DEG) ||
         coord >= (360.0f - SAFE_TOLERANCE_DEG);
}

float clampToSafeArc(int idx, float angle) {
  angle = normalizeAngle(angle);
  if (!SAFE_ARCS[idx].safeArcEnabled) return angle;
  if (isInSafeArc(idx, angle)) return angle;

  const SafeArcConfig &cfg = SAFE_ARCS[idx];
  float coord = safeArcCoordRaw(idx, angle);
  float len = safeArcLength(idx);
  float distToStart = 360.0f - coord;
  float distToEnd = coord - len;

  return (distToStart < distToEnd) ? cfg.startDeg : cfg.endDeg;
}

float safeArcCoordClamped(int idx, float angle) {
  if (!SAFE_ARCS[idx].safeArcEnabled) return normalizeAngle(angle);
  float coord = safeArcCoordRaw(idx, angle);
  float len = safeArcLength(idx);
  if (coord <= len) return coord;
  if (coord >= (360.0f - SAFE_TOLERANCE_DEG)) return 0.0f;

  float distToStart = 360.0f - coord;
  float distToEnd = coord - len;
  return (distToStart < distToEnd) ? 0.0f : len;
}


float safeArcError(int idx, float target, float current) {
  if (!SAFE_ARCS[idx].safeArcEnabled || safeArcLength(idx) >= 359.0f) {
    return angleError(target, current);
  }

  const SafeArcConfig &cfg = SAFE_ARCS[idx];
  float targetCoord = safeArcCoordClamped(idx, target);
  float currentCoord = safeArcCoordClamped(idx, current);
  return cfg.dir * (targetCoord - currentCoord);
}


float angleError(float target, float current) {
  float err = normalizeAngle(target) - normalizeAngle(current);
  while (err >  180.0f) err -= 360.0f;
  while (err < -180.0f) err += 360.0f;
  return err;
}

int signOf(float value) {
  if (value > 0.0f) return 1;
  if (value < 0.0f) return -1;
  return 0;
}


void setupMotors() {
  for (int i = 0; i < NUM_MOTORS; i++) {
    pinMode(MOTORS[i].in1, OUTPUT);
    pinMode(MOTORS[i].in2, OUTPUT);
    ledcAttach(MOTORS[i].en, PWM_FREQ, PWM_RESOLUTION);
  }
}

void setMotorSigned(int idx, int signedPwm) {
  int pwm = constrain(abs(signedPwm), 0, MAX_MOTOR_PWM);

  if (pwm == 0) {
    digitalWrite(MOTORS[idx].in1, LOW);
    digitalWrite(MOTORS[idx].in2, LOW);
  } else if (signedPwm > 0) {
    digitalWrite(MOTORS[idx].in1, HIGH);
    digitalWrite(MOTORS[idx].in2, LOW);
  } else {
    digitalWrite(MOTORS[idx].in1, LOW);
    digitalWrite(MOTORS[idx].in2, HIGH);
  }

  ledcWrite(MOTORS[idx].en, pwm);
}

void stopMotors(bool acknowledge) {
  for (int i = 0; i < NUM_MOTORS; i++) {
    setMotorSigned(i, 0);
  }

  if (acknowledge) {
    Serial.println("ACK,MOTOR,STOP");
  }
}

void printMotorAck(const char *label, const int pwms[NUM_MOTORS], bool perWheel) {
  Serial.print("ACK,MOTOR,");
  Serial.print(label);
  Serial.print(",");
  if (perWheel) {
    Serial.print("BR=");
    Serial.print(pwms[0]);
    Serial.print(",FR=");
    Serial.print(pwms[1]);
    Serial.print(",BL=");
    Serial.print(pwms[2]);
    Serial.print(",FL=");
    Serial.println(pwms[3]);
  } else {
    Serial.println(pwms[0]);
  }
}

void commandMotors(const char *label, const int pwms[NUM_MOTORS], const int signs[NUM_MOTORS], bool perWheel) {
  int appliedPwms[NUM_MOTORS];
  for (int i = 0; i < NUM_MOTORS; i++) {
    int pwm = constrain(pwms[i], 0, MAX_MOTOR_PWM);
    appliedPwms[i] = pwm;
    setMotorSigned(i, signs[i] * pwm);
  }

  printMotorAck(label, appliedPwms, perWheel);
}

void commandMotors(const char *label, int pwm, const int signs[NUM_MOTORS]) {
  int pwms[NUM_MOTORS] = {pwm, pwm, pwm, pwm};
  commandMotors(label, pwms, signs, false);
}

bool isUnsignedInteger(const String &value) {
  if (value.length() == 0) return false;
  for (int i = 0; i < value.length(); i++) {
    if (!isDigit(value.charAt(i))) return false;
  }
  return true;
}

bool parseMotorPwms(const String &text, int pwms[NUM_MOTORS], bool &perWheel) {
  perWheel = text.indexOf(',') >= 0;

  if (!perWheel) {
    if (!isUnsignedInteger(text)) return false;
    int pwm = text.toInt();
    for (int i = 0; i < NUM_MOTORS; i++) {
      pwms[i] = constrain(pwm, 0, MAX_MOTOR_PWM);
    }
    return true;
  }

  int start = 0;
  for (int i = 0; i < NUM_MOTORS; i++) {
    int comma = text.indexOf(',', start);
    String part;
    if (i == NUM_MOTORS - 1) {
      part = text.substring(start);
      if (comma >= 0) return false;
    } else {
      if (comma < 0) return false;
      part = text.substring(start, comma);
      start = comma + 1;
    }
    part.trim();
    if (!isUnsignedInteger(part)) return false;
    pwms[i] = constrain(part.toInt(), 0, MAX_MOTOR_PWM);
  }

  return true;
}

bool tryHandleMotorCommand(const String &upper) {
  if (upper == "X" || upper == "STOP") {
    stopMotors(true);
    return true;
  }

  if (upper.length() >= 2 && (upper.charAt(0) == 'F' || upper.charAt(0) == 'B' ||
                              upper.charAt(0) == 'L' || upper.charAt(0) == 'R')) {
    char dir = upper.charAt(0);
    String pwmText = upper.substring(1);
    int pwms[NUM_MOTORS];
    bool perWheel = false;
    if (!parseMotorPwms(pwmText, pwms, perWheel)) {
      Serial.println("ERR,MOTOR_PWM");
      return true;
    }

    if ((dir == 'F' || dir == 'B') && currentSwerveMode != SWERVE_INITIAL) {
      stopMotors(false);
      Serial.println("ERR,MOTOR_REJECT,FB_requires_initial_mode");
      return true;
    }

    if ((dir == 'L' || dir == 'R') && currentSwerveMode != SWERVE_PERPENDICULAR) {
      stopMotors(false);
      Serial.println("ERR,MOTOR_REJECT,LR_requires_perpendicular_mode");
      return true;
    }

    const int *signs = MOTOR_FORWARD_SIGN;
    const char *label = "F";
    if (dir == 'B') {
      signs = MOTOR_BACKWARD_SIGN;
      label = "B";
    } else if (dir == 'L') {
      signs = MOTOR_LEFT_SIGN;
      label = "L";
    } else if (dir == 'R') {
      signs = MOTOR_RIGHT_SIGN;
      label = "R";
    }

    commandMotors(label, pwms, signs, perWheel);
    return true;
  }

  bool cwPrefix = upper.startsWith("CW");
  bool ccwPrefix = upper.startsWith("CCW");
  bool cwSuffix = upper.endsWith("CW");
  bool ccwSuffix = upper.endsWith("CCW");

  if (cwPrefix || ccwPrefix || cwSuffix || ccwSuffix) {
    bool ccw = ccwPrefix || ccwSuffix;
    String pwmText;

    if (ccwPrefix) {
      pwmText = upper.substring(3);
    } else if (cwPrefix) {
      pwmText = upper.substring(2);
    } else if (ccwSuffix) {
      pwmText = upper.substring(0, upper.length() - 3);
    } else {
      pwmText = upper.substring(0, upper.length() - 2);
    }

    int pwms[NUM_MOTORS];
    bool perWheel = false;
    if (!parseMotorPwms(pwmText, pwms, perWheel)) {
      Serial.println("ERR,MOTOR_PWM");
      return true;
    }

    if (currentSwerveMode != SWERVE_MID) {
      stopMotors(false);
      Serial.println("ERR,MOTOR_REJECT,CW_CCW_requires_mid_mode");
      return true;
    }

    commandMotors(ccw ? "CCW" : "CW", pwms,
                  ccw ? MOTOR_CCW_SIGN : MOTOR_CW_SIGN, perWheel);
    return true;
  }

  return false;
}


void setupLEDs() {
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);
  setLEDs(false, false, false);
}

void setLEDs(bool red, bool green, bool blue) {
  digitalWrite(LED_RED, red ? HIGH : LOW);
  digitalWrite(LED_GREEN, green ? HIGH : LOW);
  digitalWrite(LED_BLUE, blue ? HIGH : LOW);
}

void flashLEDColor(bool red, bool green, bool blue, unsigned long durationMs) {
  setLEDs(red, green, blue);
  delay(durationMs);
  setLEDs(false, false, false);
}

void celebrateBootLEDs() {
  for (int i = 0; i < 18; i++) {
    setLEDs(true, false, false);
    delay(45);
    setLEDs(false, true, false);
    delay(45);
    setLEDs(false, false, true);
    delay(45);
    setLEDs(false, false, false);
    delay(25);
  }

  for (int i = 0; i < 14; i++) {
    setLEDs(true, true, true);
    delay(55);
    setLEDs(false, false, false);
    delay(45);
  }

  for (int i = 0; i < 12; i++) {
    setLEDs(true, true, false);
    delay(55);
    setLEDs(false, false, false);
    delay(30);
    setLEDs(false, true, true);
    delay(55);
    setLEDs(false, false, false);
    delay(30);
    setLEDs(true, false, true);
    delay(55);
    setLEDs(false, false, false);
    delay(30);
  }

  for (int i = 0; i < 10; i++) {
    setLEDs(true, false, false);
    delay(35);
    setLEDs(false, false, true);
    delay(35);
    setLEDs(false, true, false);
    delay(35);
    setLEDs(true, true, true);
    delay(45);
    setLEDs(false, false, false);
    delay(35);
  }

  for (int i = 0; i < 8; i++) {
    setLEDs(true, true, true);
    delay(70);
    setLEDs(false, false, false);
    delay(70);
  }
}

bool tryHandleLEDCommand(const String &upper) {
  if (upper == "R") {
    flashLEDColor(true, false, false, 200);
    Serial.println("ACK,LED,R");
    return true;
  }
  if (upper == "G") {
    flashLEDColor(false, true, false, 200);
    Serial.println("ACK,LED,G");
    return true;
  }
  if (upper == "B") {
    flashLEDColor(false, false, true, 200);
    Serial.println("ACK,LED,B");
    return true;
  }
  if (upper == "Z") {


    for (int i = 0; i < NUM_SERVOS; i++) {
      st.WriteSpe(IDS[i], 0, 50);
      motion[i].active = false;
    }
    stopMotors(false);
    celebrateBootLEDs();
    Serial.println("ACK,LED,Z");
    return true;
  }
  return false;
}


const int DISPENSER_SERVO_FREQ_HZ = 50;
const int DISPENSER_SERVO_MIN_US = 500;
const int DISPENSER_SERVO_MAX_US = 2500;
const int DISPENSER_SERVO_PWM_RESOLUTION = 16;
const int DISPENSER_SERVO_PERIOD_US = 1000000 / DISPENSER_SERVO_FREQ_HZ;

const int DISPENSER_NUM_TUBES = 5;
const int DISPENSER_TUBE_START[DISPENSER_NUM_TUBES] = {90, 113, 130, 150, 170};
const int DISPENSER_TUBE_END[DISPENSER_NUM_TUBES]   = {100, 123, 140, 160, 180};
const char *DISPENSER_TUBE_NAMES[DISPENSER_NUM_TUBES] = {
  "TUBE_1",
  "TUBE_2",
  "TUBE_3",
  "TUBE_4",
  "TUBE_5"
};

const int DISPENSER_DROP_START_ANGLE = 15;
const int DISPENSER_DROP_END_ANGLE = 5;
const int DISPENSER_LDR_CHECK_ANGLE = 52;
const int DISPENSER_HOME_ANGLE = DISPENSER_LDR_CHECK_ANGLE;

const int DISPENSER_LDR_SAMPLES = 7;
const int DISPENSER_LDR_BALL_PRESENT_MAX = 2850;
const int DISPENSER_LDR_NO_BALL_MIN = 3000;
const unsigned long DISPENSER_LDR_PRINT_INTERVAL_MS = 250;

const int DISPENSER_PICKUP_TRIES_PER_TUBE = 2;
const int DISPENSER_DROP_RETRY_MAX = 3;
const int DISPENSER_FAST_FINAL_APPROACH_DEG = 8;
const int DISPENSER_FAST_SETTLE_MS = 70;
const int DISPENSER_POSITION_SETTLE_MS = 45;
const int DISPENSER_CAUTIOUS_STEP_DELAY_MS = 12;
const int DISPENSER_TUBE_SWEEP_DELAY_MS = 20;
const int DISPENSER_DROP_SWEEP_DELAY_MS = 35;
const int DISPENSER_LDR_SETTLE_MS = 1000;

int dispenserCurrentAngle = DISPENSER_HOME_ANGLE;
bool dispenserServoAttached = false;
bool dispenserAutoLdr = false;
bool dispenserTubeEmpty[DISPENSER_NUM_TUBES] = {false, false, false, false, false};
unsigned long dispenserDroppedCount = 0;
unsigned long dispenserLastLdrPrintMs = 0;

int readDispenserLDR() {
  long sum = 0;
  for (int i = 0; i < DISPENSER_LDR_SAMPLES; i++) {
    sum += analogRead(DISPENSER_LDR_PIN);
    delay(3);
  }
  return (int)(sum / DISPENSER_LDR_SAMPLES);
}

bool dispenserBallPresentFromLdr(int ldr) {
  return ldr < DISPENSER_LDR_BALL_PRESENT_MAX;
}

bool dispenserNoBallFromLdr(int ldr) {
  return ldr > DISPENSER_LDR_NO_BALL_MIN;
}

void printDispenserLDR() {
  int ldr = readDispenserLDR();
  Serial.print("LDR,");
  Serial.print(ldr);
  Serial.print(",");
  if (dispenserBallPresentFromLdr(ldr)) {
    Serial.println("BALL");
  } else if (dispenserNoBallFromLdr(ldr)) {
    Serial.println("NO_BALL");
  } else {
    Serial.println("UNKNOWN");
  }
}

void writeDispenserAngle(int angle) {
  dispenserCurrentAngle = constrain(angle, 0, 180);
  if (!dispenserServoAttached) {
    dispenserServoAttached = ledcAttach(DISPENSER_SERVO_PIN,
                                        DISPENSER_SERVO_FREQ_HZ,
                                        DISPENSER_SERVO_PWM_RESOLUTION);
  }

  int pulseUs = map(dispenserCurrentAngle,
                    0,
                    180,
                    DISPENSER_SERVO_MIN_US,
                    DISPENSER_SERVO_MAX_US);
  uint32_t duty = ((uint32_t)pulseUs * (1UL << DISPENSER_SERVO_PWM_RESOLUTION)) /
                  DISPENSER_SERVO_PERIOD_US;
  ledcWrite(DISPENSER_SERVO_PIN, duty);
}

void moveDispenserCautiousTo(int target, int stepDelayMs) {
  target = constrain(target, 0, 180);
  while (dispenserCurrentAngle != target) {
    dispenserCurrentAngle += (target > dispenserCurrentAngle) ? 1 : -1;
    writeDispenserAngle(dispenserCurrentAngle);
    delay(stepDelayMs);
  }
  delay(DISPENSER_POSITION_SETTLE_MS);
}

void moveDispenserFastThenCautiousTo(int target) {
  target = constrain(target, 0, 180);
  int distance = abs(target - dispenserCurrentAngle);

  if (distance > DISPENSER_FAST_FINAL_APPROACH_DEG) {
    int dir = (target > dispenserCurrentAngle) ? 1 : -1;
    int nearTarget = target - (dir * DISPENSER_FAST_FINAL_APPROACH_DEG);
    writeDispenserAngle(nearTarget);
    delay(DISPENSER_FAST_SETTLE_MS);
  }

  moveDispenserCautiousTo(target, DISPENSER_CAUTIOUS_STEP_DELAY_MS);
}

void sweepDispenserServo(int startAngle, int endAngle, int stepDelayMs) {
  moveDispenserFastThenCautiousTo(startAngle);
  moveDispenserCautiousTo(endAngle, stepDelayMs);
}

int readDispenserLdrAtCheckPosition() {
  moveDispenserFastThenCautiousTo(DISPENSER_LDR_CHECK_ANGLE);
  delay(DISPENSER_LDR_SETTLE_MS);
  int ldr = readDispenserLDR();
  Serial.print("LDR_CHECK,");
  Serial.print(ldr);
  Serial.print(",");
  if (dispenserBallPresentFromLdr(ldr)) {
    Serial.println("BALL");
  } else if (dispenserNoBallFromLdr(ldr)) {
    Serial.println("NO_BALL");
  } else {
    Serial.println("UNKNOWN");
  }
  return ldr;
}

bool pickupFromDispenserTube(int tubeIndex) {
  Serial.print("PICKUP,");
  Serial.print(DISPENSER_TUBE_NAMES[tubeIndex]);
  Serial.print(",");
  Serial.print(DISPENSER_TUBE_START[tubeIndex]);
  Serial.print("-");
  Serial.println(DISPENSER_TUBE_END[tubeIndex]);

  sweepDispenserServo(DISPENSER_TUBE_START[tubeIndex],
                      DISPENSER_TUBE_END[tubeIndex],
                      DISPENSER_TUBE_SWEEP_DELAY_MS);

  int ldr = readDispenserLdrAtCheckPosition();
  return dispenserBallPresentFromLdr(ldr);
}

bool dropDispenserBallOnce() {
  Serial.println("DROP,BEGIN");
  sweepDispenserServo(DISPENSER_DROP_START_ANGLE,
                      DISPENSER_DROP_END_ANGLE,
                      DISPENSER_DROP_SWEEP_DELAY_MS);

  int ldr = readDispenserLdrAtCheckPosition();
  if (dispenserNoBallFromLdr(ldr)) {
    Serial.println("DROP,SUCCESS");
    return true;
  }

  Serial.println("DROP,STILL_PRESENT");
  return false;
}

bool dropDispenserBallWithRetry() {
  for (int attempt = 1; attempt <= DISPENSER_DROP_RETRY_MAX; attempt++) {
    Serial.print("DROP_ATTEMPT,");
    Serial.println(attempt);
    if (dropDispenserBallOnce()) {
      dispenserDroppedCount++;
      Serial.print("TOTAL_DROPPED,");
      Serial.println(dispenserDroppedCount);
      return true;
    }
  }

  Serial.println("DROP,FAILED");
  return false;
}

void printDispenserStatus() {
  Serial.print("STATUS,dropped=");
  Serial.print(dispenserDroppedCount);
  Serial.print(",empty=[");
  for (int i = 0; i < DISPENSER_NUM_TUBES; i++) {
    Serial.print(DISPENSER_TUBE_NAMES[i]);
    Serial.print(":");
    Serial.print(dispenserTubeEmpty[i] ? "1" : "0");
    if (i < DISPENSER_NUM_TUBES - 1) Serial.print(",");
  }
  Serial.println("]");
}

void printBoardStatus(const char *prefix) {
  Serial.print(prefix);
  Serial.print(",");
  Serial.print(FW_VERSION);
  Serial.print(",mode=");
  Serial.print(swerveModeName());
  Serial.print(",dropped=");
  Serial.print(dispenserDroppedCount);
  Serial.print(",disp_angle=");
  Serial.print(dispenserCurrentAngle);
  Serial.print(",disp_pwm=");
  Serial.print(dispenserServoAttached ? "1" : "0");
  Serial.print(",empty=[");
  for (int i = 0; i < DISPENSER_NUM_TUBES; i++) {
    Serial.print(DISPENSER_TUBE_NAMES[i]);
    Serial.print(":");
    Serial.print(dispenserTubeEmpty[i] ? "1" : "0");
    if (i < DISPENSER_NUM_TUBES - 1) Serial.print(",");
  }
  Serial.println("]");
}

void resetDispenserTubes(const char *ackLabel = "ACK,TUBE_RESET") {
  for (int i = 0; i < DISPENSER_NUM_TUBES; i++) {
    dispenserTubeEmpty[i] = false;
  }
  Serial.println(ackLabel);
  printDispenserStatus();
}

void dispenseOneBall() {
  stopMotors(false);
  Serial.println("DISPENSE,BEGIN");

  for (int tube = 0; tube < DISPENSER_NUM_TUBES; tube++) {
    if (dispenserTubeEmpty[tube]) {
      Serial.print("SKIP_EMPTY,");
      Serial.println(DISPENSER_TUBE_NAMES[tube]);
      continue;
    }

    bool gotBall = false;
    for (int attempt = 1; attempt <= DISPENSER_PICKUP_TRIES_PER_TUBE; attempt++) {
      Serial.print("PICKUP_ATTEMPT,");
      Serial.print(DISPENSER_TUBE_NAMES[tube]);
      Serial.print(",");
      Serial.println(attempt);

      if (pickupFromDispenserTube(tube)) {
        gotBall = true;
        break;
      }
    }

    if (!gotBall) {
      dispenserTubeEmpty[tube] = true;
      Serial.print("MARK_EMPTY,");
      Serial.println(DISPENSER_TUBE_NAMES[tube]);
      continue;
    }

    Serial.print("BALL_ACQUIRED,");
    Serial.println(DISPENSER_TUBE_NAMES[tube]);
    dropDispenserBallWithRetry();
    moveDispenserFastThenCautiousTo(DISPENSER_HOME_ANGLE);
    printDispenserStatus();
    Serial.println("DISPENSE,END");
    return;
  }

  moveDispenserFastThenCautiousTo(DISPENSER_HOME_ANGLE);
  Serial.println("DISPENSE,NO_BALL_AVAILABLE");
  printDispenserStatus();
}

void moveDispenserServoTo(int angle) {
  moveDispenserFastThenCautiousTo(angle);

  Serial.print("ACK,DISPENSER_SERVO,");
  Serial.print(dispenserCurrentAngle);
  Serial.print(",");
  Serial.println(readDispenserLDR());
}

void runDispenserServoSanityTest() {
  stopMotors(false);
  Serial.println("SERVO_TEST,BEGIN");

  writeDispenserAngle(0);
  Serial.println("SERVO_TEST,0");
  delay(1000);

  writeDispenserAngle(90);
  Serial.println("SERVO_TEST,90");
  delay(1000);

  writeDispenserAngle(180);
  Serial.println("SERVO_TEST,180");
  delay(1000);

  Serial.print("SERVO_TEST,END,attached=");
  Serial.print(dispenserServoAttached ? "1" : "0");
  Serial.print(",angle=");
  Serial.println(dispenserCurrentAngle);
}

bool parseDispenserAngleCommand(const String &input, int *angleOut) {
  if (input.length() == 0) return false;

  String angleText = input;
  if (input.charAt(0) == 'S') {
    angleText = input.substring(1);
    angleText.trim();
  }

  if (angleText.length() == 0) return false;
  for (int i = 0; i < angleText.length(); i++) {
    if (!isDigit(angleText.charAt(i))) return false;
  }

  int angle = angleText.toInt();
  if (angle < 0 || angle > 180) return false;

  *angleOut = angle;
  return true;
}

bool tryHandleDispenserCommand(const String &upper) {
  if (upper == "D") {
    dispenseOneBall();
    return true;
  }

  if (upper == "D0") {
    resetDispenserTubes("ACK,DISPENSER_MEMORY,TUBE_1");
    return true;
  }

  if (upper == "RESET" || upper == "TRESET") {
    resetDispenserTubes();
    return true;
  }

  if (upper == "STATUS") {
    printDispenserStatus();
    return true;
  }

  if (upper == "L") {
    printDispenserLDR();
    return true;
  }

  if (upper == "A") {
    dispenserAutoLdr = !dispenserAutoLdr;
    Serial.print("ACK,AUTO_LDR,");
    Serial.println(dispenserAutoLdr ? "ON" : "OFF");
    return true;
  }

  if (upper == "STEST" || upper == "SERVO_TEST") {
    runDispenserServoSanityTest();
    return true;
  }

  int angle = 0;
  if (parseDispenserAngleCommand(upper, &angle)) {
    moveDispenserServoTo(angle);
    return true;
  }

  return false;
}

void setupDispenser() {
  pinMode(DISPENSER_LDR_PIN, INPUT);
  dispenserServoAttached = ledcAttach(DISPENSER_SERVO_PIN,
                                      DISPENSER_SERVO_FREQ_HZ,
                                      DISPENSER_SERVO_PWM_RESOLUTION);
  writeDispenserAngle(dispenserCurrentAngle);
  delay(300);
  Serial.print("ACK,DISPENSER_ATTACH,");
  Serial.println(dispenserServoAttached ? "OK" : "FAIL");
}

void updateDispenser() {
  if (dispenserAutoLdr && millis() - dispenserLastLdrPrintMs >= DISPENSER_LDR_PRINT_INTERVAL_MS) {
    dispenserLastLdrPrintMs = millis();
    printDispenserLDR();
  }
}


void configureProfile(MotionState &m) {
  float d = fabs(m.startErr);


  m.threshold = constrain(0.20f + (d * 0.006f), 0.25f, 1.25f);
  m.minSpeed  = (int)constrain(95.0f + (d * 1.6f), 110.0f, 360.0f);
  m.maxSpeed  = (int)constrain(240.0f + (d * 14.0f), 300.0f, 2600.0f);
  m.acc       = (int)constrain(75.0f + (d * 0.45f), 75.0f, 150.0f);
}

int computeSpeed(const MotionState &m, float err) {
  float absErr = fabs(err);
  if (absErr <= m.threshold) return 0;

  float span = fabs(m.startErr) - m.threshold;
  if (span < 1.0f) span = 1.0f;

  float ratio = (absErr - m.threshold) / span;
  ratio = constrain(ratio, 0.0f, 1.0f);

  int speed = m.minSpeed + (int)((m.maxSpeed - m.minSpeed) * ratio);
  speed = constrain(speed, m.minSpeed, m.maxSpeed);
  if (err < 0.0f) speed = -speed;
  return speed;
}


void logBegin(int idx) {
  if (!SERVO_MOVE_LOGS) return;
  MotionState &m = motion[idx];
  Serial.print("TRIAL_BEGIN,id=");
  Serial.print(IDS[idx]);
  Serial.print(",fw=");
  Serial.print(FW_VERSION);
  Serial.print(",name=");
  Serial.print(SAFE_ARCS[idx].name);
  Serial.print(",requested=");
  Serial.print(m.requestedTarget, 2);
  Serial.print(",target=");
  Serial.print(m.target, 2);
  Serial.print(",start=");
  Serial.print(m.startAngle, 2);
  Serial.print(",delta=");
  Serial.print(m.startErr, 2);
  Serial.print(",threshold=");
  Serial.print(m.threshold, 2);
  Serial.print(",minSpeed=");
  Serial.print(m.minSpeed);
  Serial.print(",maxSpeed=");
  Serial.print(m.maxSpeed);
  Serial.print(",acc=");
  Serial.println(m.acc);
}

void logSample(int idx, int raw, float current, float err, int speed) {
  if (!SERVO_MOVE_LOGS) return;
  MotionState &m = motion[idx];
  Serial.print("TRIAL_SAMPLE,t=");
  Serial.print(millis() - m.startMs);
  Serial.print(",id=");
  Serial.print(IDS[idx]);
  Serial.print(",raw=");
  Serial.print(raw);
  Serial.print(",pos=");
  Serial.print(current, 2);
  Serial.print(",err=");
  Serial.print(err, 2);
  Serial.print(",speed=");
  Serial.print(speed);
  Serial.print(",acc=");
  Serial.println(m.acc);
}

void logEnd(int idx, const char *reason, int raw, float current, float err) {
  if (!SERVO_MOVE_LOGS) return;
  MotionState &m = motion[idx];
  Serial.print("TRIAL_END,id=");
  Serial.print(IDS[idx]);
  Serial.print(",reason=");
  Serial.print(reason);
  Serial.print(",fw=");
  Serial.print(FW_VERSION);
  Serial.print(",name=");
  Serial.print(SAFE_ARCS[idx].name);
  Serial.print(",requested=");
  Serial.print(m.requestedTarget, 2);
  Serial.print(",target=");
  Serial.print(m.target, 2);
  Serial.print(",raw=");
  Serial.print(raw);
  Serial.print(",final=");
  Serial.print(current, 2);
  Serial.print(",err=");
  Serial.print(err, 2);
  Serial.print(",duration=");
  Serial.print(millis() - m.startMs);
  Serial.print(",samples=");
  Serial.print(m.samples);
  Serial.print(",maxAbsErr=");
  Serial.println(m.maxAbsErr, 2);
}


void printAllPositions() {
  Serial.print("POS");
  for (int i = 0; i < NUM_SERVOS; i++) {
    Serial.print(",");
    int raw = st.ReadPos(IDS[i]);
    if (raw < 0) {
      Serial.print("ERR");
    } else {
      Serial.print(posToAngle(raw), 1);
    }
  }
  Serial.println();
}


void stopServo(int idx, const char *reason, int raw, float current, float err) {
  MotionState &m = motion[idx];
  st.WriteSpe(IDS[idx], 0, m.acc);
  m.active = false;
  m.lastSpeed = 0;
  logEnd(idx, reason, raw, current, err);


  if (IDS[idx] >= 5) {
    Serial.print("DONE,");
    Serial.print(IDS[idx]);
    Serial.print(",");
    Serial.println(reason);
  }
}

void startMove(int idx, float requestedTarget, float safeTarget, float current, float err) {
  MotionState &m = motion[idx];

  if (m.active) {
    st.WriteSpe(IDS[idx], 0, m.acc);
    Serial.print("WARN,SERVO_ABORT,");
    Serial.print(IDS[idx]);
    Serial.println(",new_command");
  }

  m.active = true;
  m.requestedTarget = normalizeAngle(requestedTarget);
  m.target = normalizeAngle(safeTarget);
  m.startAngle = current;
  m.startErr = err;
  m.lastErr = err;
  m.maxAbsErr = fabs(err);
  m.startMs = millis();
  m.lastLogMs = 0;
  m.samples = 0;
  m.stableCount = 0;
  m.lastSpeed = 0;
  configureProfile(m);

  st.WheelMode(IDS[idx]);
  delay(20);

  logBegin(idx);
}

bool commandServoTarget(int idx, float requestedTarget) {
  int id = IDS[idx];
  int raw = st.ReadPos(id);
  if (raw < 0) {
    Serial.print("ERR,READ,");
    Serial.println(id);
    return false;
  }

  float current = posToAngle(raw);
  float safeTarget = clampToSafeArc(idx, requestedTarget);

  if (SAFE_ARCS[idx].safeArcEnabled && !isInSafeArc(idx, current)) {
    Serial.print("WARN,SAFE_CURRENT_OUTSIDE,");
    Serial.print(id);
    Serial.print(",");
    Serial.print(SAFE_ARCS[idx].name);
    Serial.print(",");
    Serial.print(current, 2);
    Serial.print(",");
    Serial.print(SAFE_ARCS[idx].startDeg, 2);
    Serial.print(",");
    Serial.println(SAFE_ARCS[idx].endDeg, 2);
  }

  if (SAFE_ARCS[idx].safeArcEnabled && fabs(angleError(safeTarget, requestedTarget)) > 0.01f) {
    Serial.print("WARN,SAFE_TARGET_CLAMP,");
    Serial.print(id);
    Serial.print(",");
    Serial.print(SAFE_ARCS[idx].name);
    Serial.print(",");
    Serial.print(normalizeAngle(requestedTarget), 2);
    Serial.print(",");
    Serial.println(safeTarget, 2);
  }

  float err = safeArcError(idx, safeTarget, current);
  startMove(idx, requestedTarget, safeTarget, current, err);
  return true;
}

float swerveBaseTarget(int idx, SwerveMode mode) {
  if (mode == SWERVE_INITIAL) return SAFE_ARCS[idx].initialDeg;
  if (mode == SWERVE_MID) return SAFE_ARCS[idx].midDeg;
  if (mode == SWERVE_PERPENDICULAR) return SAFE_ARCS[idx].perpendicularDeg;
  return SAFE_ARCS[idx].initialDeg;
}

void loadLegacySwerveTargets(SwerveMode mode, float targets[NUM_SWERVE]) {
  for (int i = 0; i < NUM_SWERVE; i++) {
    targets[i] = swerveBaseTarget(WHEEL_TO_SWERVE_IDX[i], mode);
  }
}

void commandSwerveModeTargets(SwerveMode mode, const char *label, const float targets[NUM_SWERVE]) {
  stopMotors(false);
  currentSwerveMode = mode;
  int accepted = 0;
  for (int i = 0; i < NUM_SWERVE; i++) {
    currentSwerveTargets[i] = normalizeAngle(targets[i]);
    if (commandServoTarget(WHEEL_TO_SWERVE_IDX[i], currentSwerveTargets[i])) {
      accepted++;
    }
  }

  Serial.print("ACK,MODE,");
  Serial.print(label);
  Serial.print(",");
  Serial.println(accepted);
}

void commandSwerveMode(SwerveMode mode, const char *label) {
  float targets[NUM_SWERVE];
  loadLegacySwerveTargets(mode, targets);
  commandSwerveModeTargets(mode, label, targets);
}

void commandSwerveDelta(char direction, float delta) {
  if ((direction == 'R' || direction == 'L') && currentSwerveMode != SWERVE_INITIAL) {
    Serial.println("ERR,DELTA_REJECT,DR_DL_requires_initial_mode");
    return;
  }

  if ((direction == 'F' || direction == 'B') && currentSwerveMode != SWERVE_PERPENDICULAR) {
    Serial.println("ERR,DELTA_REJECT,DF_DB_requires_perpendicular_mode");
    return;
  }

  delta = constrain(delta, 0.0f, MAX_DELTA_DEG);
  const int *signs = DELTA_FORWARD_SIGN;
  if (direction == 'B') signs = DELTA_BACKWARD_SIGN;
  if (direction == 'R') signs = DELTA_RIGHT_SIGN;
  if (direction == 'L') signs = DELTA_LEFT_SIGN;

  int accepted = 0;
  for (int i = 0; i < NUM_SWERVE; i++) {
    float base = currentSwerveTargets[i];
    int servoIdx = WHEEL_TO_SWERVE_IDX[i];
    float target = normalizeAngle(base + (signs[servoIdx] * delta));
    if (commandServoTarget(servoIdx, target)) {
      accepted++;
    }
  }

  Serial.print("ACK,DELTA,");
  Serial.print(direction);
  Serial.print(",");
  Serial.print(delta, 2);
  Serial.print(",");
  Serial.println(accepted);
}

bool parseSwerveModeLabel(const String &label, SwerveMode &mode, const char **ackLabel) {
  if (label == "I" || label == "INITIAL") {
    mode = SWERVE_INITIAL;
    *ackLabel = "initial";
    return true;
  }
  if (label == "M" || label == "MID") {
    mode = SWERVE_MID;
    *ackLabel = "mid";
    return true;
  }
  if (label == "T" || label == "PERPENDICULAR") {
    mode = SWERVE_PERPENDICULAR;
    *ackLabel = "perpendicular";
    return true;
  }
  return false;
}

bool parseFloatToken(const String &text, float &value) {
  if (text.length() == 0) return false;
  char *endPtr = nullptr;
  value = strtof(text.c_str(), &endPtr);
  return endPtr != text.c_str() && *endPtr == '\0';
}

bool parseSwerveModeTargets(const String &input, SwerveMode &mode, const char **ackLabel, float targets[NUM_SWERVE]) {
  int comma1 = input.indexOf(',');
  if (comma1 < 0) return false;

  int comma2 = input.indexOf(',', comma1 + 1);
  int comma3 = input.indexOf(',', comma2 + 1);
  int comma4 = input.indexOf(',', comma3 + 1);
  int comma5 = input.indexOf(',', comma4 + 1);
  int comma6 = input.indexOf(',', comma5 + 1);
  if (comma2 < 0 || comma3 < 0 || comma4 < 0 || comma5 < 0 || comma6 >= 0) return false;

  String label = input.substring(comma1 + 1, comma2);
  label.trim();
  if (!parseSwerveModeLabel(label, mode, ackLabel)) return false;

  float br, fr, bl, fl;
  if (!parseFloatToken(input.substring(comma2 + 1, comma3), br)) return false;
  if (!parseFloatToken(input.substring(comma3 + 1, comma4), fr)) return false;
  if (!parseFloatToken(input.substring(comma4 + 1, comma5), bl)) return false;
  if (!parseFloatToken(input.substring(comma5 + 1), fl)) return false;

  if (br < 0.0f || br >= 360.0f ||
      fr < 0.0f || fr >= 360.0f ||
      bl < 0.0f || bl >= 360.0f ||
      fl < 0.0f || fl >= 360.0f) {
    return false;
  }


  targets[0] = br;
  targets[1] = fr;
  targets[2] = bl;
  targets[3] = fl;
  return true;
}


void handleCommand(String input) {
  String upper = input;
  upper.toUpperCase();

  if (upper == "P") {
    printAllPositions();
    return;
  }

  if (upper == "PING" || upper == "?") {
    printBoardStatus("PONG");
    return;
  }

  if (upper == "SYS") {
    printBoardStatus("SYS");
    return;
  }

  if (tryHandleDispenserCommand(upper)) {
    return;
  }

  if (tryHandleLEDCommand(upper)) {
    return;
  }

  if (tryHandleMotorCommand(upper)) {
    return;
  }

  if (upper.startsWith("MODE,")) {
    SwerveMode mode;
    const char *label = nullptr;
    float targets[NUM_SWERVE];
    if (!parseSwerveModeTargets(upper, mode, &label, targets)) {
      Serial.println("ERR,MODE_TARGETS");
      return;
    }
    commandSwerveModeTargets(mode, label, targets);
    return;
  }

  if (upper == "I") {
    commandSwerveMode(SWERVE_INITIAL, "initial");
    return;
  }

  if (upper == "M") {
    commandSwerveMode(SWERVE_MID, "mid");
    return;
  }

  if (upper == "T") {
    commandSwerveMode(SWERVE_PERPENDICULAR, "perpendicular");
    return;
  }

  if (upper.length() >= 3 && upper.charAt(0) == 'D' &&
      (upper.charAt(1) == 'F' || upper.charAt(1) == 'B' ||
       upper.charAt(1) == 'R' || upper.charAt(1) == 'L')) {
    float delta = upper.substring(2).toFloat();
    commandSwerveDelta(upper.charAt(1), delta);
    return;
  }

  int spaceIdx = input.indexOf(' ');
  if (spaceIdx <= 0) {
    Serial.println("ERR,CMD");
    return;
  }

  int id = input.substring(0, spaceIdx).toInt();
  float target = input.substring(spaceIdx + 1).toFloat();

  if (id < 5 || id > 9) {
    Serial.println("ERR,ID");
    return;
  }
  if (target < 0.0f || target >= 360.0f) {
    Serial.println("ERR,ANGLE");
    return;
  }

  if (commandServoTarget(id - 1, target)) {
    Serial.print("ACK,SERVO,");
    Serial.print(id);
    Serial.print(",");
    Serial.println(target, 2);
  }
}


void updateServos() {
  for (int i = 0; i < NUM_SERVOS; i++) {
    MotionState &m = motion[i];
    if (!m.active) continue;

    int raw = st.ReadPos(IDS[i]);
    if (raw < 0) {
      Serial.print("ERR,READ,");
      Serial.println(IDS[i]);
      continue;
    }

    float current = posToAngle(raw);
    float err = safeArcError(i, m.target, current);
    float absErr = fabs(err);
    m.samples++;
    if (absErr > m.maxAbsErr) m.maxAbsErr = absErr;

    if (absErr <= m.threshold) {
      m.stableCount++;
    } else {
      m.stableCount = 0;
    }

    bool closeEnough = m.stableCount >= 2;
    bool timedOut = (millis() - m.startMs) > MOVE_TIMEOUT_MS;

    int speed = computeSpeed(m, err);

    if (closeEnough) {
      stopServo(i, "threshold", raw, current, err);
      continue;
    }
    if (timedOut) {
      stopServo(i, "timeout", raw, current, err);
      continue;
    }

    st.WriteSpe(IDS[i], speed, m.acc);
    m.lastSpeed = speed;

    unsigned long now = millis();
    if (m.lastLogMs == 0 || (now - m.lastLogMs) >= LOG_PERIOD_MS) {
      logSample(i, raw, current, err, speed);
      m.lastLogMs = now;
    }

    m.lastErr = err;
  }
}


void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.setTimeout(SERIAL_TIMEOUT_MS);
  delay(300);

  Serial2.begin(1000000, SERIAL_8N1, RXD2, TXD2);
  st.pSerial = &Serial2;
  setupMotors();
  stopMotors(false);
  setupLEDs();
  celebrateBootLEDs();
  setupDispenser();

  for (int i = 0; i < NUM_SERVOS; i++) {
    st.Ping(IDS[i]);
  }
  delay(300);

  for (int i = 0; i < NUM_SERVOS; i++) {
    st.WheelMode(IDS[i]);
    delay(50);
    st.WriteSpe(IDS[i], 0, 50);
    delay(20);
  }

  for (int i = 0; i < NUM_SERVOS; i++) {
    motion[i].active = false;
    motion[i].requestedTarget = 0.0f;
    motion[i].target = 0.0f;
    motion[i].startAngle = 0.0f;
    motion[i].startErr = 0.0f;
    motion[i].lastErr = 0.0f;
    motion[i].maxAbsErr = 0.0f;
    motion[i].threshold = 0.5f;
    motion[i].minSpeed = 80;
    motion[i].maxSpeed = 180;
    motion[i].acc = 20;
    motion[i].lastSpeed = 0;
    motion[i].startMs = 0;
    motion[i].lastLogMs = 0;
    motion[i].samples = 0;
    motion[i].stableCount = 0;
  }

  Serial.print("READY,");
  Serial.println(FW_VERSION);
  printBoardStatus("SYS");
}


void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) handleCommand(input);
  }

  updateServos();
  updateDispenser();
  delay(LOOP_DELAY_MS);
}
