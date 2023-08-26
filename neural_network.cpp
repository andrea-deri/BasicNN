#include "neural_network.h"


void NeuralNetwork::init(){

  Serial.println(F("\n\n------------------------\n"));
  Serial.println(F("Arduino's neural network:"));
  Serial.print(F("Input nodes: "));
  Serial.println(INPUT_FEATURES);
  Serial.print(F("Hidden nodes: "));
  Serial.println(HIDDEN_NODES);
  Serial.print(F("Output nodes: "));
  Serial.println(OUTPUT_NODES);
  Serial.print(F("Training set samples: "));
  Serial.println(TRAINING_SET_SIZE);
  Serial.println(F("------------------------\n"));

  // initializing the random number generator
  Serial.println(F("Initializing connection weights.."));
  randomSeed(analogRead(RANDOM_SEED_INIT_PIN));

  // initializing the bias for the nodes and the weights for the input connections on hidden layer
  for (int i = 0; i < HIDDEN_NODES; i++) {
    hiddenLayerBias[i] = ((float) random(0, 1000)) / 1000.0f;
    hiddenLayer[i] = 0.0f;
    for (int j = 0; j < INPUT_FEATURES; j++) {
      inputToHiddenConnections[j][i] = ((float) random(0, 1000)) / 1000.0f;
    }
  }
  
  // initializing the bias for the nodes and the weights for the input connections on output layer
  for (int i = 0; i < OUTPUT_NODES; i++) {
    outputLayerBias[i] = ((float) random(0, 1000)) / 1000.0f;
    outputLayer[i] = 0.0f;
    for (int j = 0; j < HIDDEN_NODES; j++) {
      hiddenToOutputConnections[j][i] = ((float) random(0, 1000)) / 1000.0f;
    }
  }

  Serial.println(F("Connection weights initialization completed!"));
}


void NeuralNetwork::learn(float trainingSet[TRAINING_SET_SIZE][INPUT_FEATURES], float resultSet[TRAINING_SET_SIZE][OUTPUT_NODES]) {

  // learning started, initializing all
  Serial.println(F("Starting learning..."));
  this->error = 1.0f;
  long epoch = 1;
  
  // executing the learning loop
  for (epoch = 1; epoch < LEARNING_EPOCHS && (epoch < MIN_EPOCHS || this->error > ERROR_THRESHOLD); epoch++) { 
      
      // getting random index
      this->error = 0.0f;
      int index = random(0, TRAINING_SET_SIZE);
      
      // executing feedforwarding
      feedforwarding(trainingSet[index]);

      // then executing backpropagation of the error
      backpropagating(trainingSet[index], resultSet[index]);

      // logging after a certain amount of epochs
      if (epoch % DEBUG_LOG_AFTER_EPOCHS == 0) {
        debug(epoch, false);   
      }
  }

  // learning ended, printing values
  Serial.println(F("----------------------------------------------"));
  Serial.println(F("----------------------------------------------"));
  Serial.println(F("----------- Learning completed! --------------"));
  Serial.println(F("----------------------------------------------"));
  Serial.println(F("----------------------------------------------"));
  debug(epoch, true);
}


void NeuralNetwork::backpropagating(float input[INPUT_FEATURES], float expectedOutput[OUTPUT_NODES]) {

  // starting from output nodes, propagating backward the error. The difference
  // between the output nodes and the expected value will be used as value to be
  // multiplicated with the derivative value of the sigmoid on the current value
  // of the node.
  // Last, the calculated value will be saved in a dedicated array that will be 
  // used in order to correct the connection weights and the hidden nodes weights.
  float deltaOutput[OUTPUT_NODES];
  for (int i = 0; i < OUTPUT_NODES; i++) {
    deltaOutput[i] = (expectedOutput[i] - outputLayer[i]) * dSigmoid(outputLayer[i]);
  } 

  // backpropagating the error, saved from previous backpropagation step, on hidden node.
  // For doing so, it calculate the mean square error as the summatory of the error on single 
  // hidden nodes, weighting this error with the connection weights.
  // Then, calculate the difference between the actual weight on the hidden node and the one
  // that can fit well for the passed expected result
  float deltaHidden[HIDDEN_NODES];
  for (int i = 0; i < HIDDEN_NODES; i++) {
    float outputError = 0.0f;
    for (int j = 0; j < OUTPUT_NODES; j++) {
      outputError += deltaOutput[j] * hiddenToOutputConnections[i][j];
    }
    deltaHidden[i] = outputError * dSigmoid(hiddenLayer[i]);
    this->error += 0.5 * outputError * outputError; // this is the value calculated by the mean square error
  }

  // now, the extracted weight differences can be applied on the bias and on the connections for the output layer.
  // The new bias will be calculated adding the value of the relative delta, previously calculated, weighting
  // with the fixed learning rate.
  // The new connection weights will be calculated adding the value of the hidden node with the relative delta on the connections. 
  for (int i = 0; i < OUTPUT_NODES; i++) {
    outputLayerBias[i] += deltaOutput[i] * LEARNING_RATE;
    for (int j = 0; j < HIDDEN_NODES; j++) {
      hiddenToOutputConnections[j][i] += hiddenLayer[j] * deltaOutput[i] * LEARNING_RATE;
    }
  }
  
  // same execution will be executed on the hidden nodes, with the difference that the connection weights will not be ubdated
  // using some particular node but the input feature itself.
  for (int i = 0; i < HIDDEN_NODES; i++) {
    hiddenLayerBias[i] += deltaHidden[i] * LEARNING_RATE;
    for(int j = 0; j < INPUT_FEATURES; j++) {
      inputToHiddenConnections[j][i] += input[j] * deltaHidden[i] * LEARNING_RATE;
    }
  }
}

void NeuralNetwork::feedforwarding(float input[INPUT_FEATURES]){ 

  // updating the value for the hidden node, applying the sigmoid on the values
  // taken as input and weighted with the connection weight (from input to hidden node).
  for (int i = 0; i < HIDDEN_NODES; i++) {
    float summatory = hiddenLayerBias[i];
    for (int j = 0; j < INPUT_FEATURES; j++){
      summatory += input[j] * inputToHiddenConnections[j][i];
    }
    hiddenLayer[i] = sigmoid(summatory);
  }
      
  // updating the value for the output node, applying the sigmoid on the values
  // taken by the hidden nodes and weighted with the connection weight (from hidden to output node).
  for (int i = 0; i < OUTPUT_NODES; i++) {
    float summatory = outputLayerBias[i];
    for (int j = 0; j < HIDDEN_NODES; j++) {
      summatory += hiddenLayer[j] * hiddenToOutputConnections[j][i];
    }
    outputLayer[i] = sigmoid(summatory);
  }
}

void NeuralNetwork::predict(float input[INPUT_FEATURES], float output[OUTPUT_NODES]) {
  feedforwarding(input);
  for (int i = 0; i < OUTPUT_NODES; i ++) {
    output[i] = this->outputLayer[i];
  }
}


void NeuralNetwork::debug(long epoch, bool includeConnections) {  
  Serial.println(); 
  Serial.println(F("----------------------------------------------")); 
  Serial.print(F("Epoch: "));
  Serial.print(epoch);
  Serial.print(F("  Error = "));
  Serial.println(error, 6);
  
  Serial.println(F(" - Hidden Nodes: "));
  for(int i = 0; i < HIDDEN_NODES; i++) {       
    Serial.print(F("   #"));
    Serial.print(i);
    Serial.print(F(": "));
    Serial.print(hiddenLayer[i], 5);
    Serial.print(F(" (bias: "));
    Serial.print(hiddenLayerBias[i], 5);
    Serial.println(F(")"));
  }

  Serial.println(F(" - Output Nodes: "));
  for(int i = 0; i < OUTPUT_NODES; i++) {       
    Serial.print(F("   #"));
    Serial.print(i);
    Serial.print(F(": "));
    Serial.println(outputLayer[i], 5);
  }

  if (includeConnections) {
    Serial.println(F(" - Input-to-Hidden connections: "));
    for(int i = 0; i < INPUT_FEATURES; i++) {
      for(int j = 0; j < HIDDEN_NODES; j++) {
        Serial.print(F("   W("));
        Serial.print(i);
        Serial.print(F("-"));
        Serial.print(j);
        Serial.print(F("): "));
        Serial.println(inputToHiddenConnections[j][i], 5);
      }
    }
    Serial.println(F(" - Hidden-to-Output connections: "));
    for(int i = 0; i < HIDDEN_NODES; i++) {
      for(int j = 0; j < OUTPUT_NODES; j++) {
        Serial.print(F("   W("));
        Serial.print(i);
        Serial.print(F("-"));
        Serial.print(j);
        Serial.print(F("): "));
        Serial.println(hiddenToOutputConnections[j][i], 5);
      }
    }
  }
  Serial.println();
}


float NeuralNetwork::sigmoid(float x) {
  return 1 / (1 + exp(-x));
}

float NeuralNetwork::dSigmoid(float x) {
  return x * (1 - x);
}