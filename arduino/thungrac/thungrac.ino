#include <Servo.h>
#include <Adafruit_NeoPixel.h>

#define BUZZER_PIN 2

#define SERVO1_PIN 3
#define SERVO2_PIN 4
#define SERVO3_PIN 5

#define LED_PIN 6
#define NUM_LEDS 8

Servo servo1;
Servo servo2;
Servo servo3;

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

bool scanning = false;
unsigned long lastBeep = 0;

String buffer = "";

// config cho từng bin
int openAngle[3]  = {35, 35, 35};
int closeAngle[3] = {0,  0,  0};
int openTime[3]   = {3000, 3000, 3000};

void beep(int t = 80){
  digitalWrite(BUZZER_PIN, HIGH);
  delay(t);
  digitalWrite(BUZZER_PIN, LOW);
}

void ledWhite(){
  for(int i = 0; i < NUM_LEDS; i++){
    strip.setPixelColor(i, strip.Color(255, 255, 255));
  }
  strip.show();
}

void ledOff(){
  for(int i = 0; i < NUM_LEDS; i++){
    strip.setPixelColor(i, strip.Color(0, 0, 0));
  }
  strip.show();
}

Servo* getServo(int bin){
  if(bin == 1) return &servo1;
  if(bin == 2) return &servo2;
  if(bin == 3) return &servo3;
  return NULL;
}

int getPin(int bin){
  if(bin == 1) return SERVO1_PIN;
  if(bin == 2) return SERVO2_PIN;
  if(bin == 3) return SERVO3_PIN;
  return -1;
}

// lấy bin từ JSON kể cả có khoảng trắng sau dấu ":"
int getBin(String cmd){
  if(cmd.indexOf("\"bin\":1") != -1 || cmd.indexOf("\"bin\": 1") != -1) return 1;
  if(cmd.indexOf("\"bin\":2") != -1 || cmd.indexOf("\"bin\": 2") != -1) return 2;
  if(cmd.indexOf("\"bin\":3") != -1 || cmd.indexOf("\"bin\": 3") != -1) return 3;
  return 0;
}

void smoothMove(Servo* s, int from, int to, int stepDelay){
  if(from < to){
    for(int a = from; a <= to; a++){ s->write(a); delay(stepDelay); }
  } else {
    for(int a = from; a >= to; a--){ s->write(a); delay(stepDelay); }
  }
}

void openBin(int bin){
  Servo* s = getServo(bin);
  int pin  = getPin(bin);
  if(!s || pin == -1) return;

  int index = bin - 1;

  s->attach(pin);
  s->write(closeAngle[index]);
  delay(50);
  beep(120);
  smoothMove(s, closeAngle[index], openAngle[index], 15);
  delay(openTime[index]);
  smoothMove(s, openAngle[index], closeAngle[index], 15);
  delay(300);
  s->detach();
}

void setConfig(String cmd){
  int bin = getBin(cmd);
  if(bin == 0) return;

  int index = bin - 1;

  int openPos = cmd.indexOf("\"open\":");
  if(openPos != -1){
    int v = cmd.substring(openPos + 7).toInt();
    if(v >= 0 && v <= 180) openAngle[index] = v;
  }

  int closePos = cmd.indexOf("\"close\":");
  if(closePos != -1){
    int v = cmd.substring(closePos + 8).toInt();
    if(v >= 0 && v <= 180) closeAngle[index] = v;
  }

  int timePos = cmd.indexOf("\"time\":");
  if(timePos != -1){
    int v = cmd.substring(timePos + 7).toInt();
    if(v > 0) openTime[index] = v;
  }

  // xác nhận đã nhận config
  Serial.print("SET BIN ");
  Serial.print(bin);
  Serial.print(" open="); Serial.print(openAngle[index]);
  Serial.print(" close="); Serial.print(closeAngle[index]);
  Serial.print(" time="); Serial.println(openTime[index]);
}

void handleCommand(String cmd){
  Serial.print("CMD: ");
  Serial.println(cmd);

  if(cmd.indexOf("scan_start") != -1){
    scanning = true;
    ledWhite();
  }
  else if(cmd.indexOf("scan_end") != -1){
    scanning = false;
    ledOff();
  }
  else if(cmd.indexOf("open_bin") != -1){
    int bin = getBin(cmd);
    if(bin > 0){
      Serial.print("OPEN BIN ");
      Serial.println(bin);
      openBin(bin);
    }
  }
  // FIX: tìm giá trị "set" thay vì "cmd":"set" để tránh lỗi khoảng trắng JSON
  else if(cmd.indexOf("\"set\"") != -1){
    setConfig(cmd);
  }
  else if(cmd.indexOf("beep") != -1){
    beep();
  }
}

void setup(){
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  strip.begin();
  strip.show();
}

void loop(){
  if(scanning){
    if(millis() - lastBeep > 600){
      beep(50);
      lastBeep = millis();
    }
  }

  while(Serial.available()){
    char c = Serial.read();
    if(c == '\n'){
      buffer.trim();
      if(buffer.length() > 0){
        handleCommand(buffer);
      }
      buffer = "";
    }
    else{
      buffer += c;
    }
  }
}
