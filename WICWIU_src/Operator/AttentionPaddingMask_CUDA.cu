#ifdef __CUDNN__

#include "AttentionPaddingMask.hpp"

template class AttentionPaddingMask<float>;

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

__global__ void PaddingAttentionMaskForwardPropagate_Kernel(float *pDevInput, float *pDevMask, float *pDevOutput, int inputCapacity,
                                                         int maskCapacity, int totalCapacity, float paddingToken, int timesize, int batchsize,
                                                         int channelsize, int rowsize, int colsize) {

    int resultIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int resultShapeDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    int resultIdxDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};

    int capacityPerTime = batchsize*channelsize*rowsize*colsize;
    GetTensorDimIndex(resultIdx, resultIdxDim, capacityPerTime);

    int inputShapeDim[5] = {resultShapeDim[0], resultShapeDim[1], 1, 1, resultShapeDim[4]};
    int inputIdxDim[5]   = {resultIdxDim[0], resultIdxDim[1], 0, 0, resultIdxDim[4]};
    int maskShapeDim[5]  = {1, 1, 1, resultShapeDim[4], resultShapeDim[4]};
    int maskIdxDim[5]    = {0, 0, 0, resultIdxDim[3], resultIdxDim[4]};

    int inputIdx = GetIdx(inputShapeDim, inputIdxDim);
    int maskIdx = GetIdx(maskShapeDim, maskIdxDim);
    if (resultIdx < totalCapacity && inputIdx < inputCapacity && maskIdx < maskCapacity) {
        float fill = 1.F;
        if (pDevInput[inputIdx] == paddingToken)
            fill = 0.F;
        else
            fill = 1.F;
        pDevOutput[resultIdx] = 1 - fill * pDevMask[maskIdx];
    }
}

template <typename DTYPE> int AttentionPaddingMask<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *result = this->GetResult();

    DTYPE *m_pDevInput  = input->GetGPUData(pTime);
    DTYPE *m_pDevOutput = result->GetGPUData(pTime);
    DTYPE *m_pDevMask   = m_aSubsequentMask->GetGPUData(pTime);

    int timesize    = result->GetTimeSize();
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    int totalCapacity = result->GetCapacity() / timesize;
    int inputCapacity = input->GetCapacity() / input->GetTimeSize();
    int maskCapacity  = m_aSubsequentMask->GetCapacity() / m_aSubsequentMask->GetTimeSize();


    int threadsPerBlock = 128;
    int noBlock         = totalCapacity / threadsPerBlock + 1;
    GetKernelParameters(totalCapacity, &noBlock, &threadsPerBlock);
    PaddingAttentionMaskForwardPropagate_Kernel <<< noBlock, threadsPerBlock >>> (m_pDevInput, m_pDevMask, m_pDevOutput, inputCapacity, maskCapacity, totalCapacity,
                                                                                  m_paddingTok, timesize, batchsize, channelsize, rowsize, colsize);

    checkCudaErrors(cudaDeviceSynchronize());


    return TRUE;
}

template <typename DTYPE> int AttentionPaddingMask<DTYPE>::BackPropagateOnGPU(int pTime) {

    return TRUE;
}

#endif
