#ifdef __CUDNN__

#include "Softmax1D.hpp"

#include <assert.h>

template class Softmax1D<float>;

__device__ int *GetTensorDimIndex(int index1D, int *idxDim, int capacityPerTime) {
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        idxDim[i] = index1D/capacityPerTime;
        index1D %= capacityPerTime;
        capacityPerTime /= idxDim[i+1];
    }
    idxDim[4] = index1D;

    return idxDim;
}

__device__ int *GetTensorDimIndexWithHiddenDim(int index1D, int *idxDim, int capacityPerTime) {
    #pragma unroll
    for(int i = 0; i < 5; i++) {
        idxDim[i] = index1D/capacityPerTime;
        index1D %= capacityPerTime;
        capacityPerTime /= idxDim[i+1];
    }
    idxDim[5] = index1D;

    return idxDim;
}

__forceinline__ __device__ unsigned int GetIdx(int *shapeDim, int ti, int ba, int ch, int ro, int co) {
    return  (((ti * (shapeDim)[1] + ba) * (shapeDim)[2] + ch) * (shapeDim)[3] + ro) * (shapeDim)[4] + co;
}

__forceinline__ __device__ unsigned int GetIdx(int *shapeDim, int *idxDim) {
    return  (((idxDim[0] * (shapeDim)[1] + idxDim[1]) * (shapeDim)[2] + idxDim[2]) * (shapeDim)[3] + idxDim[3]) * (shapeDim)[4] + idxDim[4];
}

__global__ void Softmax1DExponentForwardPropagate_Kernel(float *pDevInput, float *pDevMax, float *pDevOutput, int totalCapacity,
                                                         int reducedCapacity, int dim, float epsilon, int timesize, int batchsize,
                                                         int channelsize, int rowsize, int colsize) {

    assert(dim >= 0 && dim < 5);
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int shapeDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    int idxDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};

    if (idx < totalCapacity) {
        int timePerCapacity = batchsize*channelsize*rowsize*colsize;
        GetTensorDimIndex(idx, idxDim, timePerCapacity);
    }

    shapeDim[dim] = 1;
    idxDim[dim] = 0;
    int reducedIdx = GetIdx(shapeDim, idxDim);

    if (idx < totalCapacity && reducedIdx < reducedCapacity) {
        pDevOutput[idx] = expf(pDevInput[idx] - pDevMax[reducedIdx]) + epsilon;
    }
}

__global__ void Softmax1DDivisionForwardPropagate_Kernel(float *pDevInput, float *pDevSum, float *pDevOutput, int totalCapacity,
                                                         int reducedCapacity, int dim, float epsilon, int timesize, int batchsize,
                                                         int channelsize, int rowsize, int colsize) {

    assert(dim >= 0 && dim < 5);
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int shapeDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    int idxDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};

    if (idx < totalCapacity) {
        int timePerCapacity = batchsize*channelsize*rowsize*colsize;
        GetTensorDimIndex(idx, idxDim, timePerCapacity);
    }

    shapeDim[dim] = 1;
    idxDim[dim] = 0;
    int reducedIdx = GetIdx(shapeDim, idxDim);

    if (idx < totalCapacity && reducedIdx < reducedCapacity) {
        pDevOutput[idx] = pDevInput[idx] / (pDevSum[reducedIdx] + epsilon);
    }
}

__global__ void Softmax1DBackPropagate_Kernel(float *pDevDelta, float *pDevOutput, float *pDevDeltaOutputSum, float *pDevInputDelta,
                                              int totalCapacity, int reducedCapacity, int dim, int dimSize,
                                              int timesize, int batchsize, int channelsize, int rowsize, int colsize) {

    assert(dim >= 0 && dim < 5);

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int shapeDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    int idxDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};

    if (idx < totalCapacity) {
        int timePerCapacity = batchsize * channelsize * rowsize * colsize;
        GetTensorDimIndex(idx, idxDim, timePerCapacity);
    }
    if (idx < totalCapacity) {
        int reducedShapeDim[5] = {shapeDim[0], shapeDim[1], shapeDim[2], shapeDim[3], shapeDim[4]};
        int reducedIdxDim[5] = {idxDim[0], idxDim[1], idxDim[2], idxDim[3], idxDim[4]};
        reducedShapeDim[dim] = 1; reducedIdxDim[dim] = 0;
        unsigned int reducedIdx = GetIdx(reducedShapeDim, reducedIdxDim);
        if (reducedIdx < reducedCapacity) {
            atomicAdd(&pDevInputDelta[idx], (-1 * pDevOutput[idx] * pDevDeltaOutputSum[reducedIdx]) + (pDevDelta[idx] * pDevOutput[idx]));
        }
    }
}

template <typename DTYPE> int Softmax1D<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *result = this->GetResult();

    m_pDevInput  = input->GetGPUData(pTime);
    m_pDevOutput = result->GetGPUData(pTime);
    m_pDevMax    = m_aMaxTensor->GetGPUData(pTime);
    m_pDevSum    = m_aSumTensor->GetGPUData(pTime);

    int timesize    = result->GetTimeSize();
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    checkCUDNN(cudnnReduceTensor(this->GetCudnnHandle(), m_aMaxReduceDesc,
                                 m_aDevMaxIndices, m_MaxIndicesSizeInBytes, m_aDevMaxWorkspace, m_MaxWorkspaceSizeInBytes,
                                 &m_alpha, m_aInOutTensorDesc, m_pDevInput,
                                 &m_beta, m_aMaxTensorDesc, m_pDevMax));

    Tensor<DTYPE> *exponential = Tensor<DTYPE>::Zeros(1, batchsize, channelsize, rowsize, colsize);
    exponential->SetDeviceGPU(this->GetDeviceID());
    DTYPE *pDevExp     = exponential->GetGPUData(pTime);

    int totalCapacity   = input->GetCapacity() / timesize;
    int reducedCapacity = m_aMaxTensor->GetCapacity() / timesize;

    int threadsPerBlock = 128;
    int noBlock         = totalCapacity / threadsPerBlock + 1;
    GetKernelParameters(totalCapacity, &noBlock, &threadsPerBlock);
    Softmax1DExponentForwardPropagate_Kernel<<<noBlock, threadsPerBlock>>>(m_pDevInput, m_pDevMax, pDevExp, totalCapacity, reducedCapacity,
                                                                           m_dim, m_epsilon, timesize, batchsize, channelsize, rowsize, colsize);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCUDNN(cudnnReduceTensor(this->GetCudnnHandle(), m_aNorm1ReduceDesc,
                                 m_aDevNorm1Indices, m_Norm1IndicesSizeInBytes, m_aDevNorm1Workspace, m_Norm1WorkspaceSizeInBytes,
                                 &m_alpha, m_aInOutTensorDesc, pDevExp,
                                 &m_beta, m_aSumTensorDesc, m_pDevSum));

    Softmax1DDivisionForwardPropagate_Kernel<<<noBlock, threadsPerBlock>>>(pDevExp, m_pDevSum, m_pDevOutput, totalCapacity, reducedCapacity,
                                                                           m_dim, m_epsilon, timesize, batchsize, channelsize, rowsize, colsize);

    checkCudaErrors(cudaDeviceSynchronize());
    delete exponential;
    return TRUE;
}

template <typename DTYPE> int Softmax1D<DTYPE>::BackPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *result      = this->GetResult();
    Tensor<DTYPE> *this_delta  = this->GetGradient();
    Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

    Shape *resultShape = result->GetShape();
    int dimSize = (*resultShape)[m_dim];

    m_pDevDelta       = this_delta->GetGPUData(pTime);
    m_pDevOutput      = result->GetGPUData(pTime);
    m_pDevDeltaOutput = m_aDeltaOutput->GetGPUData(pTime);
    m_pDevInputDelta  = input_delta->GetGPUData(pTime);
    m_pDevSum         = m_aSumTensor->GetGPUData(pTime);

    int timesize    = result->GetTimeSize();
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    checkCUDNN(cudnnOpTensor(this->GetCudnnHandle(), m_aOpTensorDesc,
                            &m_alpha, m_aInOutTensorDesc, m_pDevDelta,
                            &m_alpha, m_aInOutTensorDesc, m_pDevOutput,
                            &m_beta, m_aInOutTensorDesc, m_pDevDeltaOutput));

    checkCUDNN(cudnnReduceTensor(this->GetCudnnHandle(), m_aSumReduceDesc,
                                 m_aDevSumIndices, m_SumIndicesSizeInBytes, m_aDevSumWorkspace, m_SumWorkspaceSizeInBytes,
                                 &m_alpha, m_aInOutTensorDesc, m_pDevDeltaOutput,
                                 &m_beta, m_aSumTensorDesc, m_pDevSum));

    int totalCapacity   = result->GetCapacity() / timesize;
    int reducedCapacity =  m_aSumTensor->GetCapacity() / m_aSumTensor->GetTimeSize();

    int threadsPerBlock = 128;
    int noBlock         = totalCapacity / threadsPerBlock + 1;
    GetKernelParameters(totalCapacity, &noBlock, &threadsPerBlock);

    Softmax1DBackPropagate_Kernel <<< noBlock, threadsPerBlock >>>(m_pDevDelta, m_pDevOutput, m_pDevSum, m_pDevInputDelta,
                                                                     totalCapacity, reducedCapacity, m_dim, dimSize,
                                                                     timesize, batchsize, channelsize, rowsize, colsize);


    checkCudaErrors(cudaDeviceSynchronize());

    return TRUE;
}

#endif
