#ifdef __CUDNN__

#include "AdagradOptimizer.h"

// template class AdagradOptimizer<int>;
template class AdagradOptimizer<float>;
// template class AdagradOptimizer<double>;

__global__ void AdagradUpdate_kernel(float *pDevWeight, float *pDevAccGradient, int weightDim, float signed_learning_rate, float epsilon, float weightDecayRate, float *pDevGradientSquared) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim; idx += blockDim.x * gridDim.x) {
        float g = pDevAccGradient[idx];

        pDevGradientSquared[idx] = g * g;

        pDevWeight[idx]     += signed_learning_rate * weightDecayRate * pDevWeight[idx];
        pDevWeight[idx]     += signed_learning_rate / sqrt(pDevGradientSquared[idx] + epsilon) * g;
        pDevAccGradient[idx] = 0.F;
    }
}

template<typename DTYPE> int AdagradOptimizer<DTYPE>::UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pGradientSquared) {
    int noBlock = 3, threadsPerBlock = 128;

    int m_parameterDim = pParameter->GetResult()->GetCapacity();

    // GetKernelParameters(m_parameterDim, &noBlock, &threadsPerBlock);

    float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();
    float weightDecayRate = this->GetWeightDecayRate();

    Tensor<DTYPE> *trainable_data = pParameter->GetResult();
    Tensor<DTYPE> *gradient       = pParameter->GetGradient();

    DTYPE *m_pDevData                  = trainable_data->GetGPUData();
    DTYPE *m_pDevGrad                  = gradient->GetGPUData();
    DTYPE *m_pDevGradientSquared       = pGradientSquared->GetGPUData();

    AdagradUpdate_kernel << < noBlock, threadsPerBlock >> > (m_pDevData, m_pDevGrad, m_parameterDim, signed_learning_rate, m_epsilon, weightDecayRate, m_pDevGradientSquared);

    return TRUE;
}

#endif  // ifdef __CUDNN__
