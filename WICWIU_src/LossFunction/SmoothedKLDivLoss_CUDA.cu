#ifdef __CUDNN__

#include "SmoothedKLDivLoss.hpp"

template class SmoothedKLDivLoss<float>;

__forceinline__ __device__ int *GetTensorDimIndex(int index1D, int *idxDim, int capacityPerTime)
{
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        idxDim[i] = index1D/capacityPerTime;
        index1D %= capacityPerTime;
        capacityPerTime /= idxDim[i+1];
    }
    idxDim[4] = index1D;

    return idxDim;
}

__forceinline__ __device__ unsigned int GetIdx(int *shapeDim, int ti, int ba, int ch, int ro, int co) {
    return  (((ti * (shapeDim)[1] + ba) * (shapeDim)[2] + ch) * (shapeDim)[3] + ro) * (shapeDim)[4] + co;
}

__forceinline__ __device__ unsigned int GetIdx(int *shapeDim, int *idxDim) {
    return  (((idxDim[0] * (shapeDim)[1] + idxDim[1]) * (shapeDim)[2] + idxDim[2]) * (shapeDim)[3] + idxDim[3]) * (shapeDim)[4] + idxDim[4];
}

__global__ void SmoothedKLDivLoss_Forward_Kernel(float *pDevLabel, float *pDevInput, float *pDevSmoothed, float *pDevOutput, 
                                                 float smoothing, int vocabsize, int timeIndex,
                                                 int totalCapacity, int lossCapacity, int timesize, int batchsize, int channelsize, int rowsize, int colsize) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int lossDimIndex[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    GetTensorDimIndex(idx, lossDimIndex, totalCapacity);

    int lossDim[5] = {1, 1, 1, 1, 1};
    if (timeIndex == 0) {
        lossDim[0] = timesize;
        lossDim[1] = batchsize;
        lossDimIndex[2] = lossDimIndex[3] = lossDimIndex[4] = 0;
    }
    else if (timeIndex == 2) {
        lossDim[0] = timesize;
        lossDim[1] = batchsize;
        lossDim[2] = channelsize;
        lossDimIndex[3] = lossDimIndex[4] = 0;
    }

    int lossIdx = GetIdx(lossDim, lossDimIndex);    

    if (idx < totalCapacity && lossIdx < lossCapacity) {
        pDevSmoothed[idx] = (1.f - smoothing) *pDevLabel[idx] + (smoothing / (float)vocabsize);
        atomicAdd(&pDevOutput[lossIdx], pDevSmoothed[idx] * (logf(pDevSmoothed[idx]) - logf(pDevInput[idx])));
    }
}

template<typename DTYPE> Tensor<DTYPE> *SmoothedKLDivLoss<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *input  = this->GetTensor();
    Tensor<DTYPE> *label  = this->GetLabel()->GetResult();
    Tensor<DTYPE> *result = this->GetResult();

    m_pDevLabel = label->GetGPUData(pTime);
    m_pDevInput = input->GetGPUData(pTime);
    m_pDevOutput = result->GetGPUData(pTime);
    m_pDevSmoothedLabel = m_aSmoothedLabel->GetGPUData(pTime);

    int timesize    = input->GetTimeSize();
    int batchsize   = input->GetBatchSize();
    int channelsize = input->GetChannelSize();
    int rowsize     = input->GetRowSize();
    int colsize     = input->GetColSize();

    int totalCapacity = input->GetCapacity();
    int lossCapacity = result->GetCapacity();
    int threadsPerBlock = 128;
    int noBlock         = totalCapacity / threadsPerBlock + 1;

    SmoothedKLDivLoss_Forward_Kernel <<< noBlock, threadsPerBlock >>> (m_pDevLabel, m_pDevInput, m_pDevSmoothedLabel, m_pDevOutput, 
                                                                      m_Smoothing, m_VocabSize, m_timeIndex,
                                                                      totalCapacity, lossCapacity, timesize, batchsize, channelsize, rowsize, colsize);
    
    checkCudaErrors(cudaDeviceSynchronize());
    return result;
}

__global__ void SmoothedKLDivLoss_BackPropagate_kernel(float *pDevLabel, float *pDevInput, float *pDevInputDelta, 
                                                 float smoothing, int vocabsize, int totalCapacity) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < totalCapacity) {
        pDevInputDelta[idx] = -((1.f-smoothing) * pDevLabel[idx] + (smoothing/(float)vocabsize)) / pDevInput[idx];
    }
}

template<typename DTYPE> Tensor<DTYPE> *SmoothedKLDivLoss<DTYPE>::BackPropagateOnGPU(int pTime) {

    Tensor<DTYPE> *input       = this->GetTensor();
    Tensor<DTYPE> *label       = this->GetLabel()->GetResult();
    Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

    m_pDevLabel = label->GetGPUData(pTime);
    m_pDevInput = input->GetGPUData(pTime);
    m_pDevInputDelta = input_delta->GetGPUData(pTime);

    int totalCapacity = input->GetCapacity();
    int threadsPerBlock = 128;
    int noBlock         = totalCapacity / threadsPerBlock + 1;

    SmoothedKLDivLoss_BackPropagate_kernel <<< noBlock, threadsPerBlock >>> (m_pDevLabel, m_pDevInput, m_pDevInputDelta,
                                                                            m_Smoothing, m_VocabSize, totalCapacity);
    return NULL;
}

#endif  // ifdef __CUDNN__
