#ifdef __CUDNN__

#include "MaskedFill.hpp"

template class MaskedFill<float>;

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

__global__ void MaskedFillForwardPropagate_Kernel(float *pDevInput, float *pDevMask, float *pDevOutput,
                                                  int maskCapacity, int totalCapacity, float maskingValue, int timesize, int batchsize,
                                                  int channelsize, int rowsize, int colsize) {

    int resultIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int resultShapeDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    int resultIdxDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};

    int capacityPerTime = batchsize*channelsize*rowsize*colsize;
    GetTensorDimIndex(resultIdx, resultIdxDim, capacityPerTime);


    int maskShapeDim[5]  = {resultShapeDim[0], resultShapeDim[1], 1, resultShapeDim[3], resultShapeDim[4]};
    int maskIdxDim[5]    = {resultIdxDim[0], resultIdxDim[1], 0, resultIdxDim[3], resultIdxDim[4]};

    int maskIdx = GetIdx(maskShapeDim, maskIdxDim);
    if (maskIdx < maskCapacity && resultIdx < totalCapacity) {
        if (pDevMask[maskIdx])
            pDevOutput[resultIdx] = maskingValue;
        else
            pDevOutput[resultIdx] = pDevInput[resultIdx];
    }
}

__global__ void MaskedFillBackPropagate_Kernel(float *pDevDelta, float *pDevMask, float *pDevInputDelta,
                                               int maskCapacity, int totalCapacity, int timesize, int batchsize,
                                               int channelsize, int rowsize, int colsize) {


    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int shapeDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    int idxDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};

    int capacityPerTime = batchsize*channelsize*rowsize*colsize;
    GetTensorDimIndex(idx, idxDim, capacityPerTime);

    int maskShapeDim[5]  = {shapeDim[0], shapeDim[1], 1, shapeDim[3], shapeDim[4]};
    int maskIdxDim[5]    = {idxDim[0], idxDim[1], 0, idxDim[3], idxDim[4]};

    int maskIdx = GetIdx(maskShapeDim, maskIdxDim);
    if (maskIdx < maskCapacity && idx < totalCapacity) {
        if (pDevMask[maskIdx]) {
            pDevInputDelta[idx] = 0;
        }
        else {
            pDevInputDelta[idx] = pDevDelta[idx];
        }
    }
}

template <typename DTYPE> int MaskedFill<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *mask   = this->GetInput()[1]->GetResult();
    Tensor<DTYPE> *result = this->GetResult();

    DTYPE *m_pDevInput  = input->GetGPUData(pTime);
    DTYPE *m_pDevMask   = mask->GetGPUData(0);
    DTYPE *m_pDevOutput = result->GetGPUData(pTime);

    int timesize    = result->GetTimeSize();
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    int totalCapacity = result->GetCapacity() / timesize;
    int inputCapacity = input->GetCapacity() / input->GetTimeSize();
    int maskCapacity  = mask->GetCapacity() / mask->GetTimeSize();


    int threadsPerBlock = 128;
    int noBlock         = totalCapacity / threadsPerBlock + 1;
    GetKernelParameters(totalCapacity, &noBlock, &threadsPerBlock);

    MaskedFillForwardPropagate_Kernel <<< noBlock, threadsPerBlock >>> (m_pDevInput, m_pDevMask, m_pDevOutput, maskCapacity, totalCapacity,
                                                                        m_maskingValue, timesize, batchsize, channelsize, rowsize, colsize);

    checkCudaErrors(cudaDeviceSynchronize());

    return TRUE;
}

template <typename DTYPE> int MaskedFill<DTYPE>::BackPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *mask        = this->GetInput()[1]->GetResult();
    Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
    Tensor<DTYPE> *this_delta  = this->GetDelta();

    DTYPE *m_pDevInputDelta = input_delta->GetGPUData(pTime);
    DTYPE *m_pDevMask       = mask->GetGPUData(0);
    DTYPE *m_pDevDelta      = this_delta->GetGPUData(pTime);

    int timesize    = input_delta->GetTimeSize();
    int batchsize   = input_delta->GetBatchSize();
    int channelsize = input_delta->GetChannelSize();
    int rowsize     = input_delta->GetRowSize();
    int colsize     = input_delta->GetColSize();

    int totalCapacity = input_delta->GetCapacity() / timesize;
    int maskCapacity  = mask->GetCapacity() / mask->GetTimeSize();

    int threadsPerBlock = 128;
    int noBlock         = totalCapacity / threadsPerBlock + 1;
    GetKernelParameters(totalCapacity, &noBlock, &threadsPerBlock);

    MaskedFillBackPropagate_Kernel <<< noBlock, threadsPerBlock >>> (m_pDevDelta, m_pDevMask, m_pDevInputDelta, maskCapacity, totalCapacity,
                                                                     timesize, batchsize, channelsize, rowsize, colsize);

    checkCudaErrors(cudaDeviceSynchronize());

    return TRUE;
}

#endif
