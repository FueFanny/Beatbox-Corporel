#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <Arduino_FreeRTOS.h>

//imu
Adafruit_BNO055 bno = Adafruit_BNO055(55);

//ultrason
const int trigPin = 9;
const int echoPin = 10;

TaskHandle_t imuTaskHandle;
TaskHandle_t ultraTaskHandle;

//imu
void imuTask(void *pvParameters) {
  (void) pvParameters;
  for (;;) {
    imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);

    Serial.print("[IMU] Heading: ");
    Serial.print(euler.x());
    Serial.print(" Roll: ");
    Serial.print(euler.y());
    Serial.print(" Pitch: ");
    Serial.println(euler.z());
    vTaskDelay(200 / portTICK_PERIOD_MS);
  }
}

//ultrason
void ultrasonicTask(void *pvParameters) {
  (void) pvParameters;

  for (;;) {
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);

    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    long duration = pulseIn(echoPin, HIGH);
    float distance = duration * 0.034 / 2;

    Serial.print("[ULTRA] Distance: ");
    Serial.print(distance);
    Serial.println(" cm");

    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}

void setup() {
  Serial.begin(115200);

  //ultrason
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  //imu
  if (!bno.begin()) {
    Serial.println("No BNO055 detected!");
    while (1);
  }
  bno.setExtCrystalUse(true);

  xTaskCreate(
    imuTask,
    "IMU Task",
    256,
    NULL,
    1,
    &imuTaskHandle
  );
  xTaskCreate(
    ultrasonicTask,
    "Ultrasonic Task",
    256,
    NULL,
    1,
    &ultraTaskHandle
  );
}

void loop() {
  //pas utilisé avec FreeRTOS
}