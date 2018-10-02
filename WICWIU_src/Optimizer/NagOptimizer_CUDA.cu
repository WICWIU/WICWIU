#ifdef __CUDNN__

#include "NagOptimizer.h"

// template class NagOptimizer<int>;
template class NagOptimizer<float>;
// template class NagOptimizer<double>;

__global__ void NagUpdate_kernel(float *pDevWeight, float *pDevAccGradient, int weightDim, float signed_learning_rate, float momentum, float weightDecayRate, float *pDevVelocity) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim; idx += blockDim.x * gridDim.x) {
        float g = pDevAccGradient[idx];

        float pre_velo = pDevVelocity[idx];
        pDevVelocity[idx] = (momentum * pDevVelocity[idx]) + (signed_learning_rate * g);

        pDevWeight[idx]     += signed_learning_rate * weightDecayRate * pDevWeight[idx];
        pDevWeight[idx]     += -momentum * pre_velo + ((1.f + momentum) * pDevVelocity[idx]);
        pDevAccGradient[idx] = 0.F;
    }
}

template<typename DTYPE> int NagOptimizer<DTYPE>::UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity) {
    int noBlock = 3, threadsPerBlock = 128;

    int m_parameterDim = pParameter->GetResult()->GetCapacity();

    // GetKernelParameters(m_parameterDim, &noBlock, &threadsPerBlock);

    float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();
    float weightDecayRate = this->GetWeightDecayRate();

    Tensor<DTYPE> *trainable_data = pParameter->GetResult();
    Tensor<DTYPE> *gradient       = pParameter->GetGradient();

    DTYPE *m_pDevData          = trainable_data->GetGPUData();
    DTYPE *m_pDevGrad          = gradient->GetGPUData();
    DTYPE *m_pDevVelocity      = pVelocity->GetGPUData();

    NagUpdate_kernel << < noBlock, threadsPerBlock >> > (m_pDevData, m_pDevGrad, m_parameterDim, signed_learning_rate, m_momentum, weightDecayRate, m_pDevVelocity);

    return TRUE;
}

#endif  // ifdef __CUDNN__
