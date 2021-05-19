#ifndef __OPERATER_UTIL_H__
#define __OPERATER_UTIL_H__    value

#include "Operator/Tensorholder.hpp"
#include "Operator/ReShape.hpp"
#include "Operator/Concatenate.hpp"

#include "Operator/Relu.hpp"
#include "Operator/LRelu.hpp"
#include "Operator/PRelu.hpp"
#include "Operator/Sigmoid.hpp"
#include "Operator/Tanh.hpp"

#include "Operator/Add.hpp"
#include "Operator/MatMul.hpp"

#include "Operator/Convolution.hpp"
#include "Operator/TransposedConvolution.hpp"
#include "Operator/Maxpooling.hpp"
#include "Operator/Avgpooling.hpp"

#include "Operator/BatchNormalize.hpp"
// #include "Operator/CUDNNBatchNormalize.h"

#include "Operator/Softmax.hpp"
// #include "Operator/Dropout.h"

#include "Operator/NoiseGenerator/GaussianNoiseGenerator.hpp"
#include "Operator/NoiseGenerator/UniformNoiseGenerator.hpp"

#include "Operator/Switch.hpp"
#include "Operator/ReconstructionError.hpp"

//RNN관련하여 추가
#include "Operator/Recurrent.hpp"
#include "Operator/Hadamard.hpp"
#include "Operator/LSTM.hpp"
#include "Operator/Minus.hpp"
#include "Operator/GRU.hpp"

#include "Operator/Embedding.hpp"             //순서도 영향이 있음!!!!

#endif  // __OPERATER_UTIL_H__
