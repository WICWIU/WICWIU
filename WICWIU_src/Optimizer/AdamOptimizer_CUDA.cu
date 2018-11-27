#ifdef __CUDNN__

#include "AdamOptimizer.hpp"

// template class AdamOptimizer<int>;
template class AdamOptimizer<float>;
// template class AdamOptimizer<double>;

//////////////////////////////////////////////////////////////////////////////// for private method

__global__ void AdamUpdate_kernel(float *pDevWeight, float *pDevAccGradient, int weightDim, float signed_learning_rate, float beta1, float beta2, float epsilon, float weightDecayRate, float *pDevFirstMomentum, float *pDevFirstVelocity) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim; idx += blockDim.x * gridDim.x) {
        float g = pDevAccGradient[idx];
        pDevFirstMomentum[idx] = beta1 * pDevFirstMomentum[idx] + (1.F - beta1) * g;  // m (1st moment)
        pDevFirstVelocity[idx] = beta2 * pDevFirstVelocity[idx] + (1.F - beta2) * g * g;  // v (2nd moment)

        float m2 = pDevFirstMomentum[idx] / (1.F - beta1);
        float v2 = pDevFirstVelocity[idx] / (1.F - beta2);

        pDevWeight[idx]     += signed_learning_rate * weightDecayRate * pDevWeight[idx];
        pDevWeight[idx]     += signed_learning_rate / sqrt(v2 + epsilon) * m2;
        pDevAccGradient[idx] = 0.F;
    }
}

template<typename DTYPE> int AdamOptimizer<DTYPE>::UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pFirstMomentum, Tensor<DTYPE> *pFirstVelocity, Tensor<DTYPE> *pUnbiasedMomentum, Tensor<DTYPE> *pUnbiasedVelocity) {
    int noBlock = 3, threadsPerBlock = 128;

    int m_parameterDim = pParameter->GetResult()->GetCapacity();

    // GetKernelParameters(m_parameterDim, &noBlock, &threadsPerBlock);

    float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();
    float weightDecayRate = this->GetWeightDecayRate();

    Tensor<DTYPE> *trainable_data = pParameter->GetResult();
    Tensor<DTYPE> *gradient       = pParameter->GetGradient();

    DTYPE *m_pDevData          = trainable_data->GetGPUData();
    DTYPE *m_pDevGrad          = gradient->GetGPUData();
    DTYPE *m_pDevFirstMomentum = pFirstMomentum->GetGPUData();
    DTYPE *m_pDevFirstVelocity = pFirstVelocity->GetGPUData();

    AdamUpdate_kernel << < noBlock, threadsPerBlock >> > (m_pDevData, m_pDevGrad, m_parameterDim, signed_learning_rate, m_Beta1, m_Beta2, m_epsilon, weightDecayRate, m_pDevFirstMomentum, m_pDevFirstVelocity);

    return TRUE;
}

#endif  // ifdef __CUDNN__
