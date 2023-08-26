#include "neural_network.h"

NeuralNetwork net = NeuralNetwork();

void setup() {

  // init pin and other components
  Serial.begin(9600);
  pinMode(2, INPUT);
  pinMode(13, OUTPUT);
  
  // starting learning
  digitalWrite(13, HIGH);
  net.init();
  net.learn(trainingSet, resultSet);
  digitalWrite(13, LOW);
}

void loop() {

  // main loop
  if (digitalRead(2)){
    float a = ((float) map(analogRead(A0), 0, 1023, 0, 1000)) / 1000.0f;
    float b = ((float) map(analogRead(A1), 0, 1023, 0, 1000)) / 1000.0f;

    Serial.println(F("\n------------- PREDICT ------------------"));
    Serial.print(F("Input:\t"));
    Serial.print(F("A="));
    Serial.print(a);
    Serial.print(F("\tB="));
    Serial.println(b);    
    
    float input[] = {a, b};
    float output[OUTPUT_NODES];
    net.predict(input, output);
    Serial.print(F("Result:\t"));
    for (int i = 0; i < OUTPUT_NODES; i++) {
      Serial.print(F("Y("));
      Serial.print(i);
      Serial.print(F(")="));
      Serial.print(output[i]);
      Serial.print(F("\t"));
    }
    Serial.println(F("\n------------------------------------------"));
    delay(1000);
  }
}

