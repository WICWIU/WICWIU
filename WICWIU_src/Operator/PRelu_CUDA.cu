#ifdef __CUDNN__

#include "PRelu.hpp"

// template class PRelu<int>;
template class PRelu<float>;
// template class PRelu<double>;

/*!
@class PRelu cuda
*/

/*!
@brief PRelu의 ForwardPropagate 커널함수
@details ForwardPropagateOnGPU에서 호출되어 실행
@see int PRelu<DTYPE>::ForwardPropagateOnGPU(int pTime = 0)
@details 1차원으로 배열 된 block과 thread에 접근하여 연산
@param pDevInput 연산을 수행하는 input값의 GPU data
@param pDevWeight input값이 0.f 이하일 때 연산을 수행하는 weight값의 GPU data
@param pDevOutput 연산의 결과인 output값을 저장할 GPU data.
@param weightDim PRelu연산의 결과값의 dimension.
*/
__global__ void ForwardPropagate_kernel(float* pDevInput, float* pDevWeight, float* pDevOutput,
                                        int weightDim)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim;
         idx += blockDim.x * gridDim.x)
    {
        if (pDevInput[idx] > 0.f)
            pDevOutput[idx] = pDevInput[idx];
        else
            pDevOutput[idx] = pDevWeight[idx] * pDevInput[idx];
    }
}
/*!
@brief GPU에서 동작하는 ForwardPropagate 메소드.
@details GPU변수를 생성하고, 커널 함수를 실행한다.
@details noBlock는 GPU 연산시 사용되는 block의 수
@details threadsPerBlock는 한 block당 생성되는 thread개수
@details m_parameterDim는 PRelu연산의 결과값의 dimension
@details m_pDevInput, m_pDevWeight, m_pDevOutput는 GPU함수 연산에 수행되는 GPU data. 각 CPU data를
GetGPUData() 호출로 GPU data생성
@see template<typename DTYPE> DTYPE *LongArray<DTYPE>::GetGPUData(unsigned int pTime)
@details ForwardPropagate_kernel 커널 함수를 호출. 커널함수이름, 블록 수, 블록당 thread 수와 GPU
data를 다음과 같은 형식으로 호출.
@see __global__ void ForwardPropagate_kernel(float *pDevInput, float *pDevWeight, float *pDevOutput,
int weightDim)
@param pTime 연산 할 Tensor가 위치한 Time값.
@return 성공 시 TRUE.
*/
template <typename DTYPE>
int PRelu<DTYPE>::ForwardPropagateOnGPU(int pTime)
{
    int noBlock = 3, threadsPerBlock = 128;

    Tensor<DTYPE>* input = this->GetInput()[0]->GetResult();
    Tensor<DTYPE>* weight = this->GetInput()[1]->GetResult();
    Tensor<DTYPE>* result = this->GetResult();
    int m_parameterDim = this->GetResult()->GetCapacity();

    DTYPE* m_pDevInput = input->GetGPUData(pTime);
    DTYPE* m_pDevWeight = weight->GetGPUData(pTime);
    DTYPE* m_pDevOutput = result->GetGPUData(pTime);

    ForwardPropagate_kernel<<<noBlock, threadsPerBlock>>>(m_pDevInput, m_pDevWeight, m_pDevOutput,
                                                          m_parameterDim);

    return TRUE;
}

/*!
@brief PRelu의 BackPropagate 커널함수
@details BackPropagateOnGPU에서 호출되어 실행
@see int PRelu<DTYPE>::BackPropagateOnGPU(int pTime = 0)
@details 1차원으로 배열 된 block과 thread에 접근하여 연산
@param pDevInput PRelu의 input값의 GPU data
@param pDevWeight PRelu의 연산 결과인 output값이 0.f이하일 때 연산을 수행하는 weight값의 GPU data
@param pDevOutput PRelu의 연산 결과인 output값의 GPU data.
@param pDevDelta PRelu 다음 Operator의 BackPropagate 결과 값인 delta의 GPU data.
@param pDevInputDelta 연산의 결과인 delta값을 저장할 GPU data.
@param pDevWeightDelta weight의 delta값을 저장할 GPU data.
@param weightDim PRelu연산의 결과값의 dimension.
*/
__global__ void BackPropagate_kernel(float* pDevInput, float* pDevWeight, float* pDevOutput,
                                     float* pDevDelta, float* pDevInputDelta,
                                     float* pDevWeightDelta, int weightDim)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim;
         idx += blockDim.x * gridDim.x)
    {
        if (pDevOutput[idx] > 0.f)
        {
            pDevInputDelta[idx] += pDevDelta[idx];
            pDevWeightDelta[idx] += 0;
        }
        else
        {
            pDevInputDelta[idx] += pDevWeight[idx] * pDevDelta[idx];
            pDevWeightDelta[idx] += pDevInput[idx] * pDevDelta[idx];
        }
    }
}

/*!
@brief GPU에서 동작하는 BackPropagate 메소드.
@details GPU변수를 생성하고, 커널 함수를 실행한다.
@details noBlock는 GPU 연산시 사용되는 block의 수
@details threadsPerBlock는 한 block당 생성되는 thread개수
@details m_parameterDim는 PRelu연산의 결과값의 dimension
@details m_pDevInput, m_pDevWeight, m_pDevOutput, m_pDevDelta, m_pDevInputDelta, m_pDevWeightDelta는
GPU함수 연산에 수행되는 GPU data. 각 CPU data를 GetGPUData() 호출로 GPU data생성
@see template<typename DTYPE> DTYPE *LongArray<DTYPE>::GetGPUData(unsigned int pTime)
@details BackPropagate_kernel 커널 함수를 호출. 커널함수이름, 블록 수, 블록당 thread 수와 GPU data를
다음과 같은 형식으로 호출.
@see __global__ void BackPropagate_kernel(float *pDevInput, float *pDevWeight, float *pDevOutput,
float *pDevDelta, float *pDevInputDelta, float *pDevWeightDelta, int weightDim)
@param pTime 연산 할 Tensor가 위치한 Time값.
@return 성공 시 TRUE.
*/
template <typename DTYPE>
int PRelu<DTYPE>::BackPropagateOnGPU(int pTime)
{
    int noBlock = 3, threadsPerBlock = 128;

    Tensor<DTYPE>* input = this->GetInput()[0]->GetResult();
    Tensor<DTYPE>* weight = this->GetInput()[1]->GetResult();
    Tensor<DTYPE>* result = this->GetResult();
    Tensor<DTYPE>* this_delta = this->GetGradient();
    Tensor<DTYPE>* input_delta = this->GetInput()[0]->GetDelta();
    Tensor<DTYPE>* weight_delta = this->GetInput()[1]->GetDelta();
    int m_parameterDim = this->GetResult()->GetCapacity();

    DTYPE* m_pDevInput = input->GetGPUData(pTime);
    DTYPE* m_pDevWeight = weight->GetGPUData(pTime);
    DTYPE* m_pDevOutput = result->GetGPUData(pTime);

    DTYPE* m_pDevDelta = this_delta->GetGPUData(pTime);
    DTYPE* m_pDevInputDelta = input_delta->GetGPUData(pTime);
    DTYPE* m_pDevWeightDelta = weight_delta->GetGPUData(pTime);

    BackPropagate_kernel<<<noBlock, threadsPerBlock>>>(m_pDevInput, m_pDevWeight, m_pDevOutput,
                                                       m_pDevDelta, m_pDevInputDelta,
                                                       m_pDevWeightDelta, m_parameterDim);

    return TRUE;
}

#endif // ifdef __CUDNN__
