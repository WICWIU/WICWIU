#ifndef __OPERATER_UTIL_H__
#define __OPERATER_UTIL_H__ value

#include "Operator/Concatenate.hpp"
#include "Operator/ReShape.hpp"
#include "Operator/Tensorholder.hpp"

#include "Operator/LRelu.hpp"
#include "Operator/PRelu.hpp"
#include "Operator/Relu.hpp"
#include "Operator/Sigmoid.hpp"
#include "Operator/Tanh.hpp"

#include "Operator/Add.hpp"
#include "Operator/MatMul.hpp"

#include "Operator/Avgpooling.hpp"
#include "Operator/Convolution.hpp"
#include "Operator/Maxpooling.hpp"
#include "Operator/TransposedConvolution.hpp"

#include "Operator/BatchNormalize.hpp"
// #include "Operator/CUDNNBatchNormalize.h"

#include "Operator/Softmax.hpp"
// #include "Operator/Dropout.h"

#include "Operator/NoiseGenerator/GaussianNoiseGenerator.hpp"
#include "Operator/NoiseGenerator/UniformNoiseGenerator.hpp"

#include "Operator/ReconstructionError.hpp"
#include "Operator/Switch.hpp"

#include "Operator/Recurrent.hpp"

#endif // __OPERATER_UTIL_H__
