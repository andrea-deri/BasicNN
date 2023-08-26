#include "settings.h"
#include "Arduino.h"

class NeuralNetwork {

  public:

    /*
      The method permits to initialize the internal state for the neural network.
    */
    void init();

    /*
      The method permits to execute the learning phase using the passed training
      set as sample records for the process.
    */
    void learn(float trainingSet[TRAINING_SET_SIZE][INPUT_FEATURES], float resultSet[TRAINING_SET_SIZE][OUTPUT_NODES]);

    /*
      TODO: Currently not implemented
    */
    void load();

    /*
      The method permits to predict the result of certain input array following 
      the weights extracted from the learning process.
    */
    void predict(float* input, float* output);

  private:

    /*
      The method permits to get the result of the sigmoid function.
    */
    float sigmoid(float x);

    /*
      The method permits to get the result of the derivative for the sigmoid function.
    */
    float dSigmoid(float x);

    /*
      The method permits to execute a debug logging, writing all information 
      about the intermediate state of the neural network. 
    */
    void debug(long epoch, bool includeConnections);

    /*
      The method permits to execute the feedforwarding process, i.e. the action
      of trying to predict certain value based on bias and weighten connections.
    */
    void feedforwarding(float input[INPUT_FEATURES]);

    /*
      The method permits to execute the backpropagation process, i.e. the action
      of adjusting the bias and the weights of the connections and of the nodes
      propagating an error value calculated by the difference between obtained
      and expected result value.  
    */
    void backpropagating(float input[INPUT_FEATURES], float expectedOutput[OUTPUT_NODES]);


    // ---------------------------------------------------------------------------
    // -------------------------- VARIABLES --------------------------------------
    // ---------------------------------------------------------------------------

    /*
      The array that contains the abstraction of the hidden layer, where each position
      is the value assigned to the node in this layer.
    */
    float hiddenLayer[HIDDEN_NODES];

    /*
      The array that contains the bias to be applied to the corresponding node in 
      the hidden layer.
    */
    float hiddenLayerBias[HIDDEN_NODES];

    /*
      The array that contains the abstraction of the output layer, where each position
      is the value assigned to the node in this layer.
    */
    float outputLayer[OUTPUT_NODES];

    /*
      The array that contains the bias to be applied to the corresponding node in 
      the output layer.
    */
    float outputLayerBias[OUTPUT_NODES];

    /*
      The matrix that contains the weights of the connections that 
      connect input node to hidden node.
    */
    float inputToHiddenConnections[INPUT_FEATURES][HIDDEN_NODES];

    /*
      The matrix that contains the weights of the connections that 
      connect hidden node to output node.
    */
    float hiddenToOutputConnections[HIDDEN_NODES][OUTPUT_NODES];

    /*
      The value that permits to get the mean square error calculated from the 
      learning proess.
    */
    float error;
};