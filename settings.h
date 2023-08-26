// ---------------------------------------------------------
// -------------- NETWORK STRUCTURE ------------------------
// ---------------------------------------------------------

// The number of features to be passed in input
#define INPUT_FEATURES 2

// The number of nodes to be set on the output layer 
#define OUTPUT_NODES 1

// The number of nodes to be set in the single hidden layer.
// For avoiding overfitting, it is recommended to set the numebr of hidden nodes using this formula:
//
//                           (number of samples) 
//           ______________________________________________________ 
//    scaling factor * (number of input neurons + number of output neurons)
//
// where scaling factor is a value between 2 (preferred) and 10.
#define HIDDEN_NODES 2


// ---------------------------------------------------------
// -------------- LEARNING PARAMETERS ----------------------
// ---------------------------------------------------------

// The number of epochs to be executed in order to end the learning process
#define LEARNING_EPOCHS 80000L

// The minimum number of epochs to be executed in order to avoid underfitting
#define MIN_EPOCHS 40000L

// The value of the learning rate used for the learning process
#define LEARNING_RATE 0.1f

// The number of records included in the training set
#define TRAINING_SET_SIZE 4

// The minimum value that the Mean Square Error can reach during learning  
#define ERROR_THRESHOLD 0.0002f

// The amount of epochs after which the debug log will be printed on Serial port
#define DEBUG_LOG_AFTER_EPOCHS 1000


// ---------------------------------------------------------
// -------------- TRAINING SETS ----------------------------
// ---------------------------------------------------------

// The sample set used as inputs for the training process 
const float trainingSet[TRAINING_SET_SIZE][INPUT_FEATURES] = { 
  {0.0f, 0.0f},
  {1.0f, 0.0f},
  {0.0f, 1.0f},
  {1.0f, 1.0f} 
};

// The sample set used as expected output for the training process
const float resultSet[TRAINING_SET_SIZE][OUTPUT_NODES] = { 
  {0.0f},
  {1.0f},
  {1.0f},
  {0.0f} 
};


// ---------------------------------------------------------
// -------------- OTHER PARAMETERS -------------------------
// ---------------------------------------------------------

// The pin used as a seed generator for the Random Number Generator.
// This value must be not used for a connection, it must be unused for correctly perform initialization. 
#define RANDOM_SEED_INIT_PIN A3
