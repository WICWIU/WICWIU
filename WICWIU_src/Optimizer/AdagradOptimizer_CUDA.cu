#ifdef __CUDNN__

#include "AdagradOptimizer.hpp"

// template class AdagradOptimizer<int>;
template class AdagradOptimizer<float>;
// template class AdagradOptimizer<double>;

/*!
@brief 파라미터 값들을 업데이트 하는 커널함수
@details UpdateParameterOnGPU 생성자에서 호출되어 실행
@details 1차원으로 배열 된 block과 thread에 접근하여 연산
@param pDevWeight 업데이트 할 파라미터의 GPU data.
@param pDevAccGradient 업데이트 할 파라미터의 gradient.
@param weightDim 업데이트 할 파라미터의 dimension.
@param signed_learning_rate Optimizer의 학습률
@param epsilon 분모가 0이 되는 것을 방지
@param weightDecayRate 가중치 매개변수가 클 때 패널티를 부과하는 값
@see int AdagradOptimizer<DTYPE>::UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE>
*pGradientSquared)
*/
__global__ void AdagradUpdate_kernel(float* pDevWeight, float* pDevAccGradient, int weightDim,
                                     float signed_learning_rate, float epsilon,
                                     float weightDecayRate, float* pDevGradientSquared)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim;
         idx += blockDim.x * gridDim.x)
    {
        float g = pDevAccGradient[idx];

        pDevGradientSquared[idx] += g * g;

        pDevWeight[idx] += signed_learning_rate * weightDecayRate * pDevWeight[idx];
        pDevWeight[idx] += signed_learning_rate / sqrt(pDevGradientSquared[idx] + epsilon) * g;
        pDevAccGradient[idx] = 0.F;
    }
}

/*!
@brief AdagradOptimizer UpdateParameterOnGPU 생성자.
@details GPU변수를 생성하고, 커널 함수를 실행한다.
@details noBlock는 GPU 연산시 사용되는 block의 수
@details threadsPerBlock는 한 block당 생성되는 thread 갯수
@details m_parameterDim는 업데이트 할 파라미터의 dimension
@details m_pDevData, m_pDevGrad, m_pDevGradientSquared는 GPU함수 연산에 수행되는 GPU data. 각 CPU
data를 GetGPUData() 호출로 GPU data 생성
@see template<typename DTYPE> DTYPE *LongArray<DTYPE>::GetGPUData(unsigned int pTime)
@details AdagradUpdate_kernel 커널 함수를 호출. 커널함수이름, 블록 수, 블록당 thread 수와
GPU데이터를 다음과 같은 형식으로 호출.
@see __global__ void AdagradUpdate_kernel(float *pDevWeight, float *pDevAccGradient, int weightDim,
float signed_learning_rate, float epsilon, float weightDecayRate, float *pDevGradientSquared)
@param *pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
@param pGradientSquared gradient 제곱으로 업데이트 된 변수
@return 성공 시 TRUE
*/
template <typename DTYPE>
int AdagradOptimizer<DTYPE>::UpdateParameterOnGPU(Operator<DTYPE>* pParameter,
                                                  Tensor<DTYPE>* pGradientSquared)
{
    int noBlock = 3, threadsPerBlock = 128;

    int m_parameterDim = pParameter->GetResult()->GetCapacity();

    GetKernelParameters(m_parameterDim, &noBlock, &threadsPerBlock);

    float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();
    float weightDecayRate = this->GetWeightDecayRate();

    Tensor<DTYPE>* trainable_data = pParameter->GetResult();
    Tensor<DTYPE>* gradient = pParameter->GetGradient();

    DTYPE* m_pDevData = trainable_data->GetGPUData();
    DTYPE* m_pDevGrad = gradient->GetGPUData();
    DTYPE* m_pDevGradientSquared = pGradientSquared->GetGPUData();

    AdagradUpdate_kernel<<<noBlock, threadsPerBlock>>>(m_pDevData, m_pDevGrad, m_parameterDim,
                                                       signed_learning_rate, m_epsilon,
                                                       weightDecayRate, m_pDevGradientSquared);

    return TRUE;
}

#endif // ifdef __CUDNN__
