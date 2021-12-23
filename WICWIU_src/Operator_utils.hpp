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

#include "Operator/Transpose.hpp"
#include "Operator/Add.hpp"
#include "Operator/MatMul.hpp"
#include "Operator/BroadMatMul.hpp"

#include "Operator/Convolution.hpp"
#include "Operator/TransposedConvolution.hpp"
#include "Operator/Maxpooling.hpp"
#include "Operator/Avgpooling.hpp"

#include "Operator/BatchNormalize.hpp"
#include "Operator/LayerNormalize.hpp"
// #include "Operator/CUDNNBatchNormalize.h"

#include "Operator/Dropout.hpp"
#include "Operator/Softmax.hpp"
#include "Operator/Softmax1D.hpp"

#include "Operator/NoiseGenerator/GaussianNoiseGenerator.hpp"
#include "Operator/NoiseGenerator/UniformNoiseGenerator.hpp"

#include "Operator/Switch.hpp"
#include "Operator/ReconstructionError.hpp"

#include "Operator/Recurrent.hpp"
#include "Operator/Hadamard.hpp"
#include "Operator/LSTM.hpp"
#include "Operator/Minus.hpp"
#include "Operator/GRU.hpp"
#include "Operator/GRUCell.hpp"
#include "Operator/Embedding.hpp"

#include "Operator/Flip.hpp"
#include "Operator/ConcatSimilarity.hpp"

#include "Operator/Scale.hpp"
#include "Operator/LayerNormalize.hpp"
#include "Operator/MaskedFill.hpp"
#include "Operator/AttentionPaddingMask.hpp"

#include "Operator/LimitRelu.hpp"
#include "Operator/GroupedConvolution.hpp"

#endif  // __OPERATER_UTIL_H__
