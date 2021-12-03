#ifdef __CUDNN__

#include "BroadMatMul.hpp"


template class BroadMatMul<float>;

// cuBLAS

template <typename DTYPE> int BroadMatMul<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *pLeft  = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *pRight = this->GetInput()[1]->GetResult();
    Tensor<DTYPE> *result = this->GetResult();

    int timesize    = result->GetTimeSize();
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    int hiddenSize = pLeft->GetColSize();

    int leftTimeSize    = pLeft->GetTimeSize();
    int leftBatchSize   = pLeft->GetBatchSize();
    int leftChannelSize = pLeft->GetChannelSize();

    int rightTimeSize    = pRight->GetTimeSize();
    int rightBatchSize   = pRight->GetBatchSize();
    int rightChannelSize = pRight->GetChannelSize();

    m_pDevOutput = result->GetGPUData(pTime);
    if (pTime >= leftTimeSize) {
        m_pDevLeft = pLeft->GetGPUData(0);
    }
    else {
        m_pDevLeft = pLeft->GetGPUData(pTime);
    }
    if (pTime >= rightTimeSize){
        m_pDevRight = pRight->GetGPUData(0);
    }
    else {
        m_pDevRight = pRight->GetGPUData(pTime);
    }


    for (int loop = 0; loop < m_gemmLoopSize; loop ++) {

        for (int ba = 0; ba < m_gemmBatchSize; ba ++) {
            m_aOutputList[ba] = m_pDevOutput + m_gemmRowSize * m_gemmColSize * ba;
            m_aLeftList[ba]   = m_pDevLeft   + m_LeftMatrixStride  * ba;
            m_aRightList[ba]  = m_pDevRight  + m_RightMatrixStride * ba;
        }
        checkCudaErrors(cudaMemcpy(m_aDevOutputList, m_aOutputList, m_gemmBatchSize * sizeof(float *), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(m_aDevLeftList,   m_aLeftList,   m_gemmBatchSize * sizeof(float *), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(m_aDevRightList,  m_aRightList,  m_gemmBatchSize * sizeof(float *), cudaMemcpyHostToDevice));

        checkCublasErrors(cublasSgemmBatched(this->GetCublasHandle(),
                                             CUBLAS_OP_N, CUBLAS_OP_N,
                                             m_gemmColSize, m_gemmRowSize, m_gemmHidSize,
                                             &m_alpha, (const float **)m_aDevRightList, m_gemmColSize,
                                             (const float **)m_aDevLeftList, m_gemmHidSize,
                                             &m_beta, m_aDevOutputList, m_gemmColSize, m_gemmBatchSize));

        m_pDevOutput = m_pDevOutput + m_gemmRowSize * m_gemmColSize * m_gemmBatchSize;
        m_pDevLeft   = m_pDevLeft   + m_BatchedLeftMatrixStride;
        m_pDevRight  = m_pDevRight  + m_BatchedRightMatrixStride;
    }


    return TRUE;
}

template <typename DTYPE> int BroadMatMul<DTYPE>::BackPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *pLeft  = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *pRight = this->GetInput()[1]->GetResult();
    Tensor<DTYPE> *left_delta = this->GetInput()[0]->GetDelta();
    Tensor<DTYPE> *right_delta = this->GetInput()[1]->GetDelta();
    Tensor<DTYPE> *this_delta = this->GetDelta();

    // left_delta->Reset(this->GetCudnnHandle());
    // right_delta->Reset(this->GetCudnnHandle());

    int timesize    = this_delta->GetTimeSize();
    int batchsize   = this_delta->GetBatchSize();
    int channelsize = this_delta->GetChannelSize();
    int rowsize     = this_delta->GetRowSize();
    int colsize     = this_delta->GetColSize();

    int hiddenSize = pLeft->GetColSize();

    int leftTimeSize    = pLeft->GetTimeSize();
    int leftBatchSize   = pLeft->GetBatchSize();
    int leftChannelSize = pLeft->GetChannelSize();

    int rightTimeSize    = pRight->GetTimeSize();
    int rightBatchSize   = pRight->GetBatchSize();
    int rightChannelSize = pRight->GetChannelSize();

    m_pDevDelta      = this_delta->GetGPUData(pTime);
    if (pTime >= leftTimeSize) {
        m_pDevLeft   = pLeft->GetGPUData(0);
        m_pDevLeftDelta  = left_delta->GetGPUData(0);
    }
    else {
        m_pDevLeft   = pLeft->GetGPUData(pTime);
        m_pDevLeftDelta  = left_delta->GetGPUData(pTime);
    }
    if (pTime >= rightTimeSize){
        m_pDevRight  = pRight->GetGPUData(0);
        m_pDevRightDelta = right_delta->GetGPUData(0);
    }
    else {
        m_pDevRight  = pRight->GetGPUData(pTime);
        m_pDevRightDelta = right_delta->GetGPUData(pTime);
    }

    for (int loop = 0; loop < m_gemmLoopSize; loop ++) {

        for (int ba = 0; ba < m_gemmBatchSize; ba ++) {
            m_aDeltaList[ba]      = m_pDevDelta      + m_gemmRowSize * m_gemmColSize * ba;
            m_aLeftDeltaList[ba]  = m_pDevLeftDelta  + m_LeftMatrixStride  * ba;
            m_aRightDeltaList[ba] = m_pDevRightDelta + m_RightMatrixStride * ba;

            m_aLeftList[ba]  = m_pDevLeft  + m_LeftMatrixStride  * ba;
            m_aRightList[ba] = m_pDevRight + m_RightMatrixStride * ba;

            if (m_LeftMatrixStride == 0) {
                checkCublasErrors(cublasSgemm(this->GetCublasHandle(),
                                              CUBLAS_OP_T, CUBLAS_OP_N,
                                              m_gemmHidSize, m_gemmRowSize, m_gemmColSize,
                                              &m_backAlpha, m_aRightList[ba], m_gemmColSize,
                                              m_aDeltaList[ba], m_gemmColSize,
                                              &m_backBeta, m_aLeftDeltaList[ba], m_gemmHidSize));
            }
            if (m_RightMatrixStride == 0) {
                checkCublasErrors(cublasSgemm(this->GetCublasHandle(),
                                              CUBLAS_OP_N, CUBLAS_OP_T,
                                              m_gemmColSize, m_gemmHidSize, m_gemmRowSize,
                                              &m_backAlpha, m_aDeltaList[ba], m_gemmColSize,
                                              m_aLeftList[ba], m_gemmHidSize,
                                              &m_backBeta, m_aRightDeltaList[ba], m_gemmColSize));
            }
        }


        if (m_LeftMatrixStride != 0) {
            checkCudaErrors(cudaMemcpy(m_aDevDeltaList, m_aDeltaList, m_gemmBatchSize * sizeof(float *), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(m_aDevLeftDeltaList,  m_aLeftDeltaList,  m_gemmBatchSize * sizeof(float *), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(m_aDevRightList,  m_aRightList,  m_gemmBatchSize * sizeof(float *), cudaMemcpyHostToDevice));

            checkCublasErrors(cublasSgemmBatched(this->GetCublasHandle(),
                                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                                 m_gemmHidSize, m_gemmRowSize, m_gemmColSize,
                                                 &m_backAlpha, (const float **)m_aDevRightList, m_gemmColSize,
                                                 (const float **)m_aDevDeltaList, m_gemmColSize,
                                                 &m_backBeta, m_aDevLeftDeltaList, m_gemmHidSize, m_gemmBatchSize));
        }

        if (m_RightMatrixStride != 0) {
            checkCudaErrors(cudaMemcpy(m_aDevDeltaList, m_aDeltaList, m_gemmBatchSize * sizeof(float *), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(m_aDevRightDeltaList, m_aRightDeltaList, m_gemmBatchSize * sizeof(float *), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(m_aDevLeftList,   m_aLeftList,   m_gemmBatchSize * sizeof(float *), cudaMemcpyHostToDevice));

            checkCublasErrors(cublasSgemmBatched(this->GetCublasHandle(),
                                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                                 m_gemmColSize, m_gemmHidSize, m_gemmRowSize,
                                                 &m_backAlpha, (const float **)m_aDevDeltaList, m_gemmColSize,
                                                 (const float **)m_aDevLeftList, m_gemmHidSize,
                                                 &m_backBeta, m_aDevRightDeltaList, m_gemmColSize, m_gemmBatchSize));
        }

        m_pDevLeft  = m_pDevLeft  + m_BatchedLeftMatrixStride;
        m_pDevRight = m_pDevRight + m_BatchedRightMatrixStride;

        m_pDevDelta      = m_pDevDelta      + m_gemmRowSize * m_gemmColSize * m_gemmBatchSize;
        m_pDevLeftDelta  = m_pDevLeftDelta  + m_BatchedLeftMatrixStride;
        m_pDevRightDelta = m_pDevRightDelta + m_BatchedRightMatrixStride;

    }


    return TRUE;
}


template class BahdanauBroadMatMul<float>;

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

__global__ void BahdanauBroadMatMulForwardPropagate_Kernel(float *pDevOutput, float *pDevLeft, float *pDevRight, int resultCapacity, int leftCapacity, int rightCapacity,
                                                   int timesize, int batchsize, int channelsize, int rowsize, int colsize, int hiddenSize,
                                                   int leftTimeSize, int leftBatchSize, int leftChannelSize,
                                                   int rightTimeSize, int rightBatchSize, int rightChannelSize) {

    int resultIdx = threadIdx.x + blockDim.x * blockIdx.x;

    int resultIdxDim[5] = {0, batchsize, channelsize, rowsize, colsize};
    GetTensorDimIndex(resultIdx, resultIdxDim, resultCapacity);

    int leftShapeDim[5] = {1, leftBatchSize, leftChannelSize, rowsize, hiddenSize};
    int leftIdxDim[5] = {0, 0, 0, 0, 0};

    int rightShapeDim[5] = {1, rightBatchSize, rightChannelSize, hiddenSize, colsize};
    int rightIdxDim[5] = {0, 0, 0, 0, 0};

    #pragma unroll
    for (int i = 0; i < 5; i ++) {
        leftIdxDim[i] = fminf(resultIdxDim[i], leftShapeDim[i]-1);
        rightIdxDim[i] = fminf(resultIdxDim[i], rightShapeDim[i]-1);
    }
    leftIdxDim[4] = 0;
    rightIdxDim[3] = 0;

    if (resultIdx < resultCapacity) {
        #pragma unroll
        for (int i = 0; i < hiddenSize; i++) {
            unsigned int leftIdx = GetIdx(leftShapeDim, leftIdxDim);
            unsigned int rightIdx = GetIdx(rightShapeDim, rightIdxDim);
            if (leftIdx < leftCapacity && rightIdx < rightCapacity) {
                atomicAdd(&pDevOutput[resultIdx], pDevLeft[leftIdx] * pDevRight[rightIdx]);
            }
            leftIdxDim[4] += 1;
            rightIdxDim[3] += 1;
        }
    }
}

__global__ void BahdanauBroadMatMulBackPropagate_Kernel(float *pDevDelta, float *pDevLeft, float *pDevRight, float *pDevLeftDelta, float *pDevRightDelta, int resultCapacity, int leftCapacity, int rightCapacity,
                                                   int timesize, int batchsize, int channelsize, int rowsize, int colsize, int hiddenSize,
                                                   int leftTimeSize, int leftBatchSize, int leftChannelSize,
                                                   int rightTimeSize, int rightBatchSize, int rightChannelSize) {

    int resultIdx = threadIdx.x + blockDim.x * blockIdx.x;

    int resultIdxDim[5] = {0, batchsize, channelsize, rowsize, colsize};
    GetTensorDimIndex(resultIdx, resultIdxDim, resultCapacity);

    int leftShapeDim[5] = {1, leftBatchSize, leftChannelSize, rowsize, hiddenSize};
    int leftIdxDim[5] = {0, 0, 0, 0, 0};

    int rightShapeDim[5] = {1, rightBatchSize, rightChannelSize, hiddenSize, colsize};
    int rightIdxDim[5] = {0, 0, 0, 0, 0};

    #pragma unroll
    for (int i = 0; i < 5; i ++) {
        leftIdxDim[i] = fminf(resultIdxDim[i], leftShapeDim[i]-1);
        rightIdxDim[i] = fminf(resultIdxDim[i], rightShapeDim[i]-1);
    }
    leftIdxDim[4] = 0;
    rightIdxDim[3] = 0;

    if (resultIdx < resultCapacity) {
        #pragma unroll
        for (int i = 0; i < hiddenSize; i++) {
            unsigned int leftIdx = GetIdx(leftShapeDim, leftIdxDim);
            unsigned int rightIdx = GetIdx(rightShapeDim, rightIdxDim);
            if (leftIdx < leftCapacity && rightIdx < rightCapacity) {
                atomicAdd(&pDevLeftDelta[leftIdx], pDevRight[rightIdx] * pDevDelta[resultIdx]);
                atomicAdd(&pDevRightDelta[rightIdx], pDevLeft[leftIdx] * pDevDelta[resultIdx]);
            }
            leftIdxDim[4] += 1;
            rightIdxDim[3] += 1;
        }
    }
}

template <typename DTYPE> int BahdanauBroadMatMul<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *pLeft  = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *pRight = this->GetInput()[1]->GetResult();
    Tensor<DTYPE> *result = this->GetResult();

    int timesize    = result->GetTimeSize();
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    int hiddenSize = pLeft->GetColSize();

    int leftTimeSize    = pLeft->GetTimeSize();
    int leftBatchSize   = pLeft->GetBatchSize();
    int leftChannelSize = pLeft->GetChannelSize();

    int rightTimeSize    = pRight->GetTimeSize();
    int rightBatchSize   = pRight->GetBatchSize();
    int rightChannelSize = pRight->GetChannelSize();

    m_pDevOutput = result->GetGPUData(pTime);
    if (pTime >= leftTimeSize) {
        m_pDevLeft   = pLeft->GetGPUData(0);
    }
    else {
        m_pDevLeft   = pLeft->GetGPUData(pTime);
    }
    if (pTime >= rightTimeSize){
        m_pDevRight  = pRight->GetGPUData(0);
    }
    else {
        m_pDevRight  = pRight->GetGPUData(pTime);
    }

    int resultCapacity = result->GetCapacity()/timesize;
    int leftCapacity = pLeft->GetCapacity()/leftTimeSize;
    int rightCapacity = pRight->GetCapacity()/rightTimeSize;

    int threadsPerBlock = 128;
    int noBlock         = resultCapacity / threadsPerBlock + 1;
    GetKernelParameters(resultCapacity, &noBlock, &threadsPerBlock);

    BahdanauBroadMatMulForwardPropagate_Kernel <<< noBlock, threadsPerBlock >>>(m_pDevOutput, m_pDevLeft, m_pDevRight, resultCapacity, leftCapacity, rightCapacity,
                                               timesize, batchsize, channelsize, rowsize, colsize, hiddenSize,
                                               leftTimeSize, leftBatchSize, leftChannelSize,
                                               rightTimeSize, rightBatchSize, rightChannelSize);
    checkCudaErrors(cudaDeviceSynchronize());

    return TRUE;
}

template <typename DTYPE> int BahdanauBroadMatMul<DTYPE>::BackPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *pLeft  = this->GetInput()[0]->GetResult();
    Tensor<DTYPE> *pRight = this->GetInput()[1]->GetResult();
    Tensor<DTYPE> *left_delta = this->GetInput()[0]->GetDelta();
    Tensor<DTYPE> *right_delta = this->GetInput()[1]->GetDelta();
    Tensor<DTYPE> *this_delta = this->GetDelta();

    int timesize    = this_delta->GetTimeSize();
    int batchsize   = this_delta->GetBatchSize();
    int channelsize = this_delta->GetChannelSize();
    int rowsize     = this_delta->GetRowSize();
    int colsize     = this_delta->GetColSize();

    int hiddenSize = pLeft->GetColSize();

    int leftTimeSize    = pLeft->GetTimeSize();
    int leftBatchSize   = pLeft->GetBatchSize();
    int leftChannelSize = pLeft->GetChannelSize();

    int rightTimeSize    = pRight->GetTimeSize();
    int rightBatchSize   = pRight->GetBatchSize();
    int rightChannelSize = pRight->GetChannelSize();

    if (pTime >= leftTimeSize) {
        m_pDevLeft   = pLeft->GetGPUData(0);
        m_pDevLeftDelta  = left_delta->GetGPUData(0);
    }
    else {
        m_pDevLeft   = pLeft->GetGPUData(pTime);
        m_pDevLeftDelta  = left_delta->GetGPUData(pTime);
    }
    if (pTime >= rightTimeSize){
        m_pDevRight  = pRight->GetGPUData(0);
        m_pDevRightDelta = right_delta->GetGPUData(0);
    }
    else {
        m_pDevRight  = pRight->GetGPUData(pTime);
        m_pDevRightDelta = right_delta->GetGPUData(pTime);
    }
    m_pDevDelta      = this_delta->GetGPUData(pTime);

    int resultCapacity = this_delta->GetCapacity()/timesize;
    int leftCapacity = pLeft->GetCapacity()/leftTimeSize;
    int rightCapacity = pRight->GetCapacity()/rightTimeSize;

    int threadsPerBlock = 128;
    int noBlock         = resultCapacity / threadsPerBlock + 1;
    GetKernelParameters(resultCapacity, &noBlock, &threadsPerBlock);

    BahdanauBroadMatMulBackPropagate_Kernel <<< noBlock, threadsPerBlock >>>(m_pDevDelta, m_pDevLeft, m_pDevRight, m_pDevLeftDelta, m_pDevRightDelta, resultCapacity, leftCapacity, rightCapacity,
                                               timesize, batchsize, channelsize, rowsize, colsize, hiddenSize,
                                               leftTimeSize, leftBatchSize, leftChannelSize,
                                               rightTimeSize, rightBatchSize, rightChannelSize);
    checkCudaErrors(cudaDeviceSynchronize());
    return TRUE;
}

#endif
