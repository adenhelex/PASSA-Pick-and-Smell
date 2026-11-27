#include <Arduino.h>
#include <Servo.h>
#include <Wire.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

// Wrist pins follow the supplied RBAE02 example for Arduino Uno.
// M6 is the cylinder. The UART name is "cylinder". There is no jaw.
// Servo 10° points the opening down. Servo 73° parks the cup sideways.
// Those two angles are dry-run assumptions, not a measured horn calibration.
constexpr byte ENABLE_PIN = 12;
constexpr unsigned long STEP_MS = 12;
constexpr byte COVER_SAMPLE = 10;
constexpr byte COVER_PARK = 73;
enum JointId : byte { BASE, SHOULDER, ELBOW, WRIST_VERTICAL, WRIST_ROTATION, CYLINDER };

struct Joint {
  const char *name;
  byte pin;
  byte minimum;
  byte maximum;
  byte home;
};

const Joint joints[] = {
  {"base",           11, 0, 180,  90},
  {"shoulder",       10, 15, 165,  45},
  {"elbow",           9, 0, 180, 180},
  {"wrist_vertical",  6, 0, 180, 180},
  {"wrist_rotation",  5, 0, 180,  90},
  {"cylinder",        3, COVER_SAMPLE, COVER_PARK, COVER_PARK}
};

// Uno A4/A5 are shared with SDA/SCL.
const byte inputPins[] = {A0, A1, A2, A3, A4, A5};

constexpr byte JOINT_COUNT = sizeof(joints) / sizeof(joints[0]);
Servo servos[JOINT_COUNT];
byte angle[JOINT_COUNT];
byte target[JOINT_COUNT];
unsigned long lastStep;
bool moving = false;
char line[64];
byte length = 0;
bool overflow = false;

void printStatus() {
  Serial.print(moving ? F("STATUS moving") : F("STATUS idle"));
  for (byte i = 0; i < JOINT_COUNT; ++i) {
    Serial.print(' ');
    Serial.print(joints[i].name);
    Serial.print('=');
    Serial.print(angle[i]);
  }
  Serial.println();
}

void startMove() {
  moving = memcmp(angle, target, sizeof(angle)) != 0;
  lastStep = millis();
  if (moving) Serial.println(F("OK moving"));
  else Serial.println(F("DONE"));
}

void updateMotion() {
  const unsigned long now = millis();
  if (!moving || now - lastStep < STEP_MS) return;
  lastStep = now;
  bool remaining = false;
  for (byte i = 0; i < JOINT_COUNT; ++i) {
    if (angle[i] == target[i]) continue;
    // The cylinder has two positions. It does not stop between them.
    if (i == CYLINDER) angle[i] = target[i];
    else if (angle[i] < target[i]) ++angle[i];
    else --angle[i];
    servos[i].write(angle[i]);
    if (angle[i] != target[i]) remaining = true;
  }
  if (!remaining) {
    moving = false;
    Serial.println(F("DONE"));
  }
}

void scanI2c() {
  Wire.begin();
  Wire.setWireTimeout(3000, true);
  bool found = false;
  for (byte address = 0x08; address <= 0x77; ++address) {
    Wire.beginTransmission(address);
    const byte result = Wire.endTransmission();
    if (Wire.getWireTimeoutFlag()) {
      Wire.clearWireTimeoutFlag();
      Wire.end();
      Serial.println(F("ERR i2c timeout"));
      return;
    }
    if (result == 0) {
      found = true;
      Serial.print(F("I2C 0x"));
      if (address < 0x10) Serial.print('0');
      Serial.println(address, HEX);
    }
  }
  Wire.end();
  if (!found) Serial.println(F("I2C no devices"));
  Serial.println(F("DONE"));
}

void handleCommand() {
  for (char *c = line; *c; ++c) *c = tolower(static_cast<unsigned char>(*c));
  const char *delimiters = " \t";
  char *name = strtok(line, delimiters);
  if (!name) return;
  char *action = strtok(nullptr, delimiters);
  char *argument = strtok(nullptr, delimiters);
  if (strtok(nullptr, delimiters)) {
    Serial.println(F("ERR too many words"));
    return;
  }

  if (!action && !strcmp(name, "pins")) {
    Serial.println(F("INPUTS I0=A0(14) I1=A1(15) I2=A2(16) I3=A3(17) I4=A4(18) I5=A5(19)"));
    Serial.println(F("TWI SDA=A4 SCL=A5 | UART RX0=D0 TX0=D1"));
    return;
  }
  if (name[0] == 'i' && name[1] >= '0' && name[1] <= '5' && !name[2]) {
    if (!action || argument || (strcmp(action, "read") && strcmp(action, "digital"))) {
      Serial.println(F("ERR use <i0-i5> read or <i0-i5> digital"));
      return;
    }
    const byte pin = inputPins[name[1] - '0'];
    const int value = !strcmp(action, "read") ? analogRead(pin) : digitalRead(pin);
    Serial.print(F("INPUT "));
    Serial.print(name);
    Serial.print('=');
    Serial.println(value);
    return;
  }
  if (!action && !strcmp(name, "status")) {
    printStatus();
    return;
  }
  if (!action && !strcmp(name, "help")) {
    Serial.println(F("COMMANDS <joint> rotate <degrees> | cylinder park | cylinder sample | home | stop | status"));
    Serial.println(F("JOINTS base shoulder elbow wrist_vertical wrist_rotation cylinder"));
    Serial.println(F("CYLINDER sample=10 opens down. park=73 rests sideways. No jaw."));
    Serial.println(F("I/O <i0-i5> read | <i0-i5> digital | i2c scan | pins"));
    return;
  }
  if (!action && !strcmp(name, "stop")) {
    memcpy(target, angle, sizeof(angle));
    moving = false;
    Serial.println(F("OK stopped"));
    return;
  }
  if (moving) {
    Serial.println(F("ERR busy; wait for DONE or send stop"));
    return;
  }
  if (action && !argument && !strcmp(name, "i2c") && !strcmp(action, "scan")) {
    scanI2c();
    return;
  }
  if (!action && !strcmp(name, "home")) {
    for (byte i = 0; i < JOINT_COUNT; ++i) target[i] = joints[i].home;
    startMove();
    return;
  }

  if (!strcmp(name, "gripper")) {
    Serial.println(F("ERR the joint is cylinder; use cylinder rotate, park, or sample"));
    return;
  }

  byte joint = 0;
  while (joint < JOINT_COUNT && strcmp(name, joints[joint].name)) ++joint;
  if (joint == JOINT_COUNT || !action) {
    Serial.println(F("ERR unknown command; send help"));
    return;
  }

  long degrees;
  if (!strcmp(action, "rotate") && argument) {
    char *end;
    degrees = strtol(argument, &end, 10);
    if (end == argument || *end) {
      Serial.println(F("ERR angle must be a whole number"));
      return;
    }
  } else if (joint == CYLINDER && !argument && !strcmp(action, "park")) {
    degrees = COVER_PARK;
  } else if (joint == CYLINDER && !argument && !strcmp(action, "sample")) {
    degrees = COVER_SAMPLE;
  } else if (joint == CYLINDER && (!strcmp(action, "grab") || !strcmp(action, "grap") ||
             !strcmp(action, "close") || !strcmp(action, "release") || !strcmp(action, "open"))) {
    Serial.println(F("ERR the cylinder has no jaw; use cylinder rotate, park, or sample"));
    return;
  } else {
    Serial.println(F("ERR use <joint> rotate <degrees>"));
    return;
  }

  const Joint &axis = joints[joint];
  if (joint == CYLINDER && degrees != COVER_SAMPLE && degrees != COVER_PARK) {
    Serial.println(F("ERR cylinder is 10 or 73 only"));
    return;
  }
  if (degrees < axis.minimum || degrees > axis.maximum) {
    Serial.print(F("ERR angle range "));
    Serial.print(axis.minimum);
    Serial.print(F(" to "));
    Serial.println(axis.maximum);
    return;
  }
  target[joint] = static_cast<byte>(degrees);
  startMove();
}

void setup() {
  Serial.begin(115200);  // Uno UART on D0/RX and D1/TX, also shared with USB serial.
  for (byte pin : inputPins) pinMode(pin, INPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);
  for (byte i = 0; i < JOINT_COUNT; ++i) {
    angle[i] = target[i] = joints[i].home;
    servos[i].write(angle[i]);
    servos[i].attach(joints[i].pin);
  }
  digitalWrite(ENABLE_PIN, HIGH);
  delay(700);
  Serial.println(F("READY Braccio UART"));
  printStatus();
}

void loop() {
  updateMotion();
  // Limit serial processing per loop to keep motion updates responsive.
  for (byte count = 0; count < 32 && Serial.available(); ++count) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\r' || c == '\n') {
      if (overflow) Serial.println(F("ERR command too long"));
      else if (length) {
        line[length] = '\0';
        handleCommand();
      }
      length = 0;
      overflow = false;
    } else if (!overflow) {
      if (length < sizeof(line) - 1) line[length++] = c;
      else overflow = true;
    }
  }
}
