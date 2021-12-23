#ifdef __CUDNN__

#include "LayerNormalize.hpp"

template class LayerNormalize<float>;

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

__global__ void GetVariance_kernel(float *mean, float *sqrtOfMeanSquared, float *var, int layerSize, int weightDim) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < weightDim) {
        var[idx] = powf(sqrtOfMeanSquared[idx], 2)/layerSize - powf(mean[idx], 2);
    }
}

__global__ void Normalize_kernel(float *input, float *normalized, float *mean, float *var, int batchIndex, float epsilon, int timesize, int batchsize, int channelsize, int rowsize, int colsize, int weightDim) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int shapeDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    int inputDimIndex[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    GetTensorDimIndex(idx, inputDimIndex, weightDim);

    for (int i = batchIndex + 1; i < 5; i++) {
        shapeDim[i] = 1;
        inputDimIndex[i] = 0;
    }
   
    int reducedIdx = GetIdx(shapeDim, inputDimIndex);

    if(idx < weightDim) {
        normalized[idx] = (input[idx]-mean[reducedIdx])/(sqrtf(var[reducedIdx] + epsilon) + epsilon);
    }

}

__global__ void AffineTransform_kernel(float *normalized, float *output, float *scale, float *bias, int batchIndex, int timesize, int batchsize, int channelsize, int rowsize, int colsize, int weightDim) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int shapeDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    int inputDimIndex[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    GetTensorDimIndex(idx, inputDimIndex, weightDim);

    for(int i = 0; i <= batchIndex; i++){
        shapeDim[i] = 1;
        inputDimIndex[i] = 0;
    }
   
    int statisticIdx = GetIdx(shapeDim, inputDimIndex);

    if(idx < weightDim)
        output[idx] = scale[statisticIdx]*normalized[idx] + bias[statisticIdx];
}


template<typename DTYPE> int LayerNormalize<DTYPE>::ComputeLayerStatisticsOnGPU() {
    Tensor<DTYPE> *result = this->GetResult();
    checkCUDNN(cudnnReduceTensor(this->GetCudnnHandle(), meanReduceTenDesc,
                                 meanIndicesSpace, meanIndicesSize, meanWorkSpace, meanWorkSpaceSize,
                                 &m_alpha, inputTenDesc, m_pDevInput,
                                 &m_beta, meanTenDesc, m_pDevMean));

    checkCUDNN(cudnnReduceTensor(this->GetCudnnHandle(), sqrtOfMeanSquaredReduceTenSesc, 
                                 sqrtOfMeanSquareIndicesSpace, sqrtOfMeanSquareIndicesSize, sqrtOfMeanSquaredWorkSpace, sqrtOfMeanSquaredWorkSpaceSize, 
                                 &m_alpha, inputTenDesc, m_pDevInput, 
                                 &m_beta, sqrtOfMeanSquaredTenDesc, m_pDevSqrtOfSquaredSum));

    int layerSize = 1;
    for (int i = m_batchIndex + 1; i < 5; i++) {
        layerSize *= result->GetDim(i);
    }

    int totalThread = m_pSqrtOfSquaredSumTensor->GetCapacity();
    int noBlock, threadsPerBlock;
    GetKernelParameters(totalThread, &noBlock, &threadsPerBlock);

    GetVariance_kernel<<<noBlock,threadsPerBlock>>>(m_pDevMean, m_pDevSqrtOfSquaredSum, m_pDevVar, layerSize, totalThread);
    checkCudaErrors(cudaDeviceSynchronize());
    return TRUE;
}

template<typename DTYPE> int LayerNormalize<DTYPE>::NormalizeOnGPU() {
    Tensor<DTYPE> *result = this->GetResult();

    int timesize    = result->GetTimeSize();
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    int totalThread = result->GetCapacity();
    int noBlock, threadsPerBlock;
    GetKernelParameters(totalThread, &noBlock, &threadsPerBlock);

    Normalize_kernel<<<noBlock,threadsPerBlock>>>(m_pDevInput, m_pDevCachedNormalized, m_pDevMean, m_pDevVar, m_batchIndex, m_epsilon, timesize, batchsize, channelsize, rowsize, colsize, totalThread);

    return TRUE;
}

template<typename DTYPE> int LayerNormalize<DTYPE>::AffineTransformOnGPU() {
    Tensor<DTYPE> *result = this->GetResult();
   
    int timesize    = result->GetTimeSize();
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    int totalThread = result->GetCapacity();
    int noBlock, threadsPerBlock;
    GetKernelParameters(totalThread, &noBlock, &threadsPerBlock);

    AffineTransform_kernel<<<noBlock,threadsPerBlock>>>(m_pDevCachedNormalized, m_pDevOutput, m_pDevScale, m_pDevBias, m_batchIndex, timesize, batchsize, channelsize, rowsize, colsize, totalThread);

    return TRUE;
}


template<typename DTYPE> int LayerNormalize<DTYPE>::ForwardPropagateOnGPU(int pTime) {

    Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *scale = this->GetInput()[1]->GetResult();
    Tensor<DTYPE> *bias = this->GetInput()[2]->GetResult();
    Tensor<DTYPE> *result = this->GetResult();

    m_pDevInput = input->GetGPUData(pTime);
    m_pDevOutput = result->GetGPUData(pTime);
    m_pDevCachedNormalized = m_pCachedNormalizedTensor->GetGPUData(pTime);
    m_pDevMean = m_pMeanTensor->GetGPUData(pTime);
    m_pDevSqrtOfSquaredSum = m_pSqrtOfSquaredSumTensor->GetGPUData(pTime);
    m_pDevMeanSquared = m_pMeanSquaredTensor->GetGPUData(pTime);
    m_pDevVar = m_pVarTensor->GetGPUData(pTime);
    m_pDevScale = scale->GetGPUData(pTime);
    m_pDevBias = bias->GetGPUData(pTime);

    ComputeLayerStatisticsOnGPU();

    NormalizeOnGPU();

    AffineTransformOnGPU();

    return TRUE;
}



__global__ void LayerNormalize_AffineBackPropagate_Kernel(float *pDevDelta, float *pDevCachedNormalized, float *pDevScale, float *pDevScaleDelta, float *pDevBiasDelta, float *pDevXHatDelta, float *pDevXHatScaledDelta,
                                                          int batchIndex, int totalCapacity, int affineCapacity, int timesize, int batchsize, int channelsize, int rowsize, int colsize) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int shapeDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    int inputDimIndex[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    GetTensorDimIndex(idx, inputDimIndex, totalCapacity);

    for (int i = 0; i <= batchIndex; i++) {
        shapeDim[i] = 1;
        inputDimIndex[i] = 0;
    }
    int affineIdx = GetIdx(shapeDim, inputDimIndex);

    if (idx < totalCapacity && affineIdx < affineCapacity) {
        atomicAdd(&pDevBiasDelta[affineIdx], pDevDelta[idx]);
        atomicAdd(&pDevScaleDelta[affineIdx], pDevDelta[idx] * pDevCachedNormalized[idx]);
        atomicAdd(&pDevXHatDelta[idx], pDevDelta[idx] * pDevScale[affineIdx]);
        atomicAdd(&pDevXHatScaledDelta[idx], pDevXHatDelta[idx] * pDevCachedNormalized[idx]);
    }

}

__global__ void LayerNormalize_NormalizeBackPropagate_Kernel(float *pXHatDelta, float *pDevXHatAvgDelta, float *pDevXHatScaleAvgDelta, float *pDevCachedNormalized, float *pDevVar, float *pDevInputDelta,
                                                             int batchIndex, int n, int epsilon, int totalCapacity, int reducedCapacity, int timesize, int batchsize, int channelsize, int rowsize, int colsize) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int shapeDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    int inputDimIndex[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    GetTensorDimIndex(idx, inputDimIndex, totalCapacity);

    for (int i = batchIndex + 1; i < 5; i++) {
        shapeDim[i] = 1;
        inputDimIndex[i] = 0;
    }
   
    int reducedIdx = GetIdx(shapeDim, inputDimIndex);

    if (idx < totalCapacity && reducedIdx < reducedCapacity) {
        pDevInputDelta[idx] = 1.f / ((n-1) * sqrtf(pDevVar[reducedIdx] + epsilon)) * ((n-1) * pXHatDelta[idx] - pDevXHatAvgDelta[reducedIdx]*(n-1) - (pDevXHatScaleAvgDelta[reducedIdx]*(n-1) * pDevCachedNormalized[idx]));
    }
}

template<typename DTYPE> int LayerNormalize<DTYPE>::BackPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *this_delta   = this->GetDelta();
    Tensor<DTYPE> *input_delta  = this->GetInput()[0]->GetDelta();
    Tensor<DTYPE> *scale_delta  = this->GetInput()[1]->GetDelta();
    Tensor<DTYPE> *scale_result = this->GetInput()[1]->GetResult();
    Tensor<DTYPE> *bias_delta   = this->GetInput()[2]->GetDelta();
    
    Shape *pShape             = this_delta->GetShape();
    Shape *pAffineShape       = scale_delta->GetShape();
    Shape *pLayerSummaryShape = m_pMeanTensor->GetShape();

    int n = 1;

    for (int i = m_batchIndex + 1; i < 5; i ++) {
        n *= this_delta->GetShape()->GetDim(i);
    }

    m_pDevDelta = this_delta->GetGPUData(pTime);
    m_pDevInputDelta = input_delta->GetGPUData(pTime);
    m_pDevScaleDelta = scale_delta->GetGPUData(pTime);
    m_pDevBiasDelta = bias_delta->GetGPUData(pTime);
    m_pDevScale = scale_result->GetGPUData(pTime);
    m_pDevVar = m_pVarTensor->GetGPUData(pTime);
    m_pDevCachedNormalized = m_pCachedNormalizedTensor->GetGPUData(pTime);

    int totalCapacity = this_delta->GetCapacity();
    int affineCapacity = scale_delta->GetCapacity();
    int reducedCapacity = m_pVarTensor->GetCapacity();

    int threadsPerBlock = 128;
    int noBlock         = totalCapacity / threadsPerBlock + 1;
    GetKernelParameters(totalCapacity ,&noBlock, &threadsPerBlock);

    int timesize    = this_delta->GetTimeSize();
    int batchsize   = this_delta->GetBatchSize();
    int channelsize = this_delta->GetChannelSize();
    int rowsize     = this_delta->GetRowSize();
    int colsize     = this_delta->GetColSize();

    m_pDevXHatDelta         = m_pXHat_delta->GetGPUData(pTime);
    m_pDevXHatScaleDelta    = m_pXHatScaled_delta->GetGPUData(pTime);
    m_pDevXHatAvgDelta      = m_pXHatAvg_delta->GetGPUData(pTime);
    m_pDevXHatScaleAvgDelta = m_pXHatScaledAvg_delta->GetGPUData(pTime);

    checkCudaErrors(cudaMemset(m_pDevXHatDelta, 0, sizeof(float) * totalCapacity));
    checkCudaErrors(cudaMemset(m_pDevXHatScaleDelta, 0, sizeof(float) * totalCapacity));
    checkCudaErrors(cudaMemset(m_pDevXHatAvgDelta, 0, sizeof(float) * reducedCapacity));
    checkCudaErrors(cudaMemset(m_pDevXHatScaleAvgDelta, 0, sizeof(float) * reducedCapacity));


    LayerNormalize_AffineBackPropagate_Kernel<<<noBlock, threadsPerBlock>>>(m_pDevDelta, m_pDevCachedNormalized, m_pDevScale, m_pDevScaleDelta, m_pDevBiasDelta, m_pDevXHatDelta, m_pDevXHatScaleDelta,
                                                                            m_batchIndex, totalCapacity, affineCapacity, timesize, batchsize, channelsize, rowsize, colsize);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCUDNN(cudnnReduceTensor(this->GetCudnnHandle(), meanReduceTenDesc,
                                 meanIndicesSpace, meanIndicesSize, meanWorkSpace, meanWorkSpaceSize,
                                 &m_alpha, inputTenDesc, m_pDevXHatDelta,
                                 &m_beta, meanTenDesc, m_pDevXHatAvgDelta));

    checkCUDNN(cudnnReduceTensor(this->GetCudnnHandle(), meanReduceTenDesc,
                                 meanIndicesSpace, meanIndicesSize, meanWorkSpace, meanWorkSpaceSize,
                                 &m_alpha, inputTenDesc, m_pDevXHatScaleDelta,
                                 &m_beta, meanTenDesc, m_pDevXHatScaleAvgDelta));

    LayerNormalize_NormalizeBackPropagate_Kernel<<<noBlock, threadsPerBlock>>>(m_pDevXHatDelta, m_pDevXHatAvgDelta, m_pDevXHatScaleAvgDelta, m_pDevCachedNormalized, m_pDevVar, m_pDevInputDelta,
                                                                               m_batchIndex, n, m_epsilon, totalCapacity, reducedCapacity, timesize, batchsize, channelsize, rowsize, colsize);
    checkCudaErrors(cudaDeviceSynchronize());

    return TRUE;
}
#endif
