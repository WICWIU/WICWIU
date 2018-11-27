#ifdef __CUDNN__

#include "RMSPropOptimizer.hpp"

// template class RMSPropOptimizer<int>;
template class RMSPropOptimizer<float>;
// template class RMSPropOptimizer<double>;

//////////////////////////////////////////////////////////////////////////////// for private method

__global__ void RMSPropUpdate_kernel(float *pDevWeight, float *pDevAccGradient, int weightDim, float signed_learning_rate, float decay, float epsilon, float weightDecayRate, float *pMeanSquared) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim; idx += blockDim.x * gridDim.x) {
        float g = pDevAccGradient[idx];
        pMeanSquared[idx] = (decay * pMeanSquared[idx]) + ((1.F - decay) * (g * g)); //meansquared

        pDevWeight[idx]     += signed_learning_rate * weightDecayRate * pDevWeight[idx];
        pDevWeight[idx]     += signed_learning_rate / sqrt(pMeanSquared[idx] + epsilon) * g;
        pDevAccGradient[idx] = 0.F;
    }
}

__global__ void RMSPropUpdate_kernelForCentered(float *pDevWeight, float *pDevAccGradient, int weightDim, float signed_learning_rate, float decay, float epsilon, float weightDecayRate, float *pMeanSquared, float *pMeanGrad) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim; idx += blockDim.x * gridDim.x) {
        float g = pDevAccGradient[idx];
        pMeanGrad[idx] = (decay * pMeanGrad[idx]) + ((1.f - decay) * g); //meangrad
        pMeanSquared[idx] = (decay * pMeanSquared[idx]) + ((1.F - decay) * (g * g)); //meansquared

        pDevWeight[idx]     += signed_learning_rate * weightDecayRate * pDevWeight[idx];
        pDevWeight[idx]     += signed_learning_rate / sqrt((pMeanSquared[idx] - (pMeanGrad[idx] * pMeanGrad[idx])) + epsilon) * g;

        pDevAccGradient[idx] = 0.F;
    }
}

template<typename DTYPE> int RMSPropOptimizer<DTYPE>::UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pMeanSquared) {
    int noBlock = 3, threadsPerBlock = 128;

    int m_parameterDim = pParameter->GetResult()->GetCapacity();

    // GetKernelParameters(m_parameterDim, &noBlock, &threadsPerBlock);

    float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();
    float weightDecayRate = this->GetWeightDecayRate();

    Tensor<DTYPE> *trainable_data = pParameter->GetResult();
    Tensor<DTYPE> *gradient       = pParameter->GetGradient();

    DTYPE *m_pDevData          = trainable_data->GetGPUData();
    DTYPE *m_pDevGrad          = gradient->GetGPUData();
    DTYPE *m_pDevMeanSquared   = pMeanSquared->GetGPUData();

    RMSPropUpdate_kernel << < noBlock, threadsPerBlock >> > (m_pDevData, m_pDevGrad, m_parameterDim, signed_learning_rate, m_decay, m_epsilon, weightDecayRate, m_pDevMeanSquared);

    return TRUE;
}

template<typename DTYPE> int RMSPropOptimizer<DTYPE>::UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pMeanSquared, Tensor<DTYPE> *pMeanGrad) {
    int noBlock = 3, threadsPerBlock = 128;

    int m_parameterDim = pParameter->GetResult()->GetCapacity();

    // GetKernelParameters(m_parameterDim, &noBlock, &threadsPerBlock);

    float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();
    float weightDecayRate = this->GetWeightDecayRate();

    Tensor<DTYPE> *trainable_data = pParameter->GetResult();
    Tensor<DTYPE> *gradient       = pParameter->GetGradient();

    DTYPE *m_pDevData          = trainable_data->GetGPUData();
    DTYPE *m_pDevGrad          = gradient->GetGPUData();
    DTYPE *m_pDevMeanSquared   = pMeanSquared->GetGPUData();
    DTYPE *m_pDevMeanGrad      = pMeanGrad->GetGPUData();

    RMSPropUpdate_kernelForCentered << < noBlock, threadsPerBlock >> > (m_pDevData, m_pDevGrad, m_parameterDim, signed_learning_rate, m_decay, m_epsilon, weightDecayRate, m_pDevMeanSquared, m_pDevMeanGrad);

    return TRUE;
}

#endif  // ifdef __CUDNN__
