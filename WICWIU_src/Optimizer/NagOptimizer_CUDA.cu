#ifdef __CUDNN__

#include "NagOptimizer.hpp"

// template class NagOptimizer<int>;
template class NagOptimizer<float>;
// template class NagOptimizer<double>;

/*!
@brief 파라미터 값들을 업데이트 하는 커널함수.
@details UpdateParameterOnGPU 생성자에서 호출되어 실행.
@details 1차원으로 배열 된 block과 thread에 접근하여 연산.
@param pDevWeight 업데이트 할 파라미터의 GPU data.
@param pDevAccGradient 업데이트 할 파라미터의 gradient.
@param weightDim 업데이트 할 파라미터의 dimension.
@param signed_learning_rate Optimizer의 학습률.
@param momentum step size 조정 값.
@param weightDecayRate 가중치 매개변수가 클 때 패널티를 부과하는 값.
@param pDevVelocity 업데이트 될 pDevVelocity
@see  int NagOptimizer<DTYPE>::UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity)
*/
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

/*!
@brief NagOptimizer UpdateParameterOnGPU 생성자.
@details GPU변수를 생성하고, 커널 함수를 실행한다.
@details noBlock는 GPU 연산시 사용되는 block의 수
@details threadsPerBlock는 한 block당 생성되는 thread 갯수
@details m_parameterDim는 업데이트 할 파라미터의 dimension
@details m_pDevData, m_pDevGrad, m_pDevGradientSquared는 GPU함수 연산에 수행되는 GPU data. 각 CPU data를 GetGPUData() 호출로 GPU data 생성
@see template<typename DTYPE> DTYPE *LongArray<DTYPE>::GetGPUData(unsigned int pTime)
@details AdagradUpdate_kernel 커널 함수를 호출. 커널함수이름, 블록 수, 블록당 thread 수와 GPU데이터를 다음과 같은 형식으로 호출.
@see __global__ void NagUpdate_kernel(float *pDevWeight, float *pDevAccGradient, int weightDim, float signed_learning_rate, float momentum, float weightDecayRate, float *pDevVelocity)
@param *pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
@param pVelocity 업데이트할 pVelocity
@return 성공 시 TRUE
*/
template<typename DTYPE> int NagOptimizer<DTYPE>::UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity) {
    int noBlock = 3, threadsPerBlock = 128;

    int m_parameterDim = pParameter->GetResult()->GetCapacity();

    GetKernelParameters(m_parameterDim, &noBlock, &threadsPerBlock);

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
