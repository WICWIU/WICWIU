#ifdef __CUDNN__

#include "Concatenate.h"

// template class ConcatenateChannelWise<int>;
template class ConcatenateChannelWise<float>;
// template class ConcatenateChannelWise<double>;

__global__ void ConcatenateChannelWise_ForwardPropagate_kernel(int sizeOfResultImg, int sizeOfInputImg, int timesize, int batchsize, float *result, float *input, int preSize) {
    int indexOfResult = 0;
    int indexOfInput  = 0;
    int cnt           = 0;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeOfInputImg; idx += blockDim.x * gridDim.x) {
        for (int ba = 0; ba < batchsize; ba++) {
            indexOfResult = ba * sizeOfResultImg + (preSize + blockIdx.x + cnt) * blockDim.x + threadIdx.x;
            indexOfInput  = ba * sizeOfInputImg + idx;

            result[indexOfResult] = input[indexOfInput];
            // result[0] = input[0];
        }

        cnt += gridDim.x;
    }
}

template<typename DTYPE> int ConcatenateChannelWise<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *result = this->GetResult();
    Tensor<DTYPE> *input  = NULL;

    int timesize    = result->GetTimeSize();
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    Shape *resultTenShape = result->GetShape();

    int noBlock         = 0;
    int threadsPerBlock = rowsize * colsize;
    int sizeOfPlane     = threadsPerBlock;
    int sizeOfResultImg = channelsize * sizeOfPlane;
    int sizeOfInputImg  = 0;

    DTYPE *result_gpu = result->GetGPUData();
    DTYPE *input_gpu  = NULL;

    int preSize  = 0;

    for (int opnum = 0; opnum < m_noOperator; opnum++) {
        input          = this->GetInput()[opnum]->GetResult();
        input_gpu      = input->GetGPUData();
        noBlock        = input->GetChannelSize();
        preSize        = m_aAccumulate[opnum];
        sizeOfInputImg = noBlock * sizeOfPlane;

        ConcatenateChannelWise_ForwardPropagate_kernel << < noBlock, threadsPerBlock >> > (sizeOfResultImg, sizeOfInputImg, timesize, batchsize, result_gpu, input_gpu, preSize);
    }

    // this->ForwardPropagate();

    return TRUE;
}

__global__ void ConcatenateChannelWise_BackPropagate_kernel(int sizeOfResultImg, int sizeOfInputImg, int timesize, int batchsize, float *delta_gpu, float *input_delta_gpu, int preSize) {
    int indexOfResult = 0;
    int indexOfInput  = 0;
    int cnt           = 0;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeOfInputImg; idx += blockDim.x * gridDim.x) {
        for (int ba = 0; ba < batchsize; ba++) {
            indexOfResult = ba * sizeOfResultImg + (preSize + blockIdx.x + cnt) * blockDim.x + threadIdx.x;
            indexOfInput  = ba * sizeOfInputImg + idx;

            input_delta_gpu[indexOfInput] = delta_gpu[indexOfResult];
            // input_delta_gpu[0] = delta_gpu[0];
        }

        cnt += gridDim.x;
    }
}

template<typename DTYPE> int ConcatenateChannelWise<DTYPE>::BackPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *this_delta  = this->GetDelta();
    Tensor<DTYPE> *input_delta = NULL;

    int timesize    = this_delta->GetTimeSize();
    int batchsize   = this_delta->GetBatchSize();
    int channelsize = this_delta->GetChannelSize();
    int rowsize     = this_delta->GetRowSize();
    int colsize     = this_delta->GetColSize();

    Shape *resultTenShape = this_delta->GetShape();

    int noBlock         = 0;
    int threadsPerBlock = rowsize * colsize;
    int sizeOfPlane     = threadsPerBlock;
    int sizeOfResultImg = channelsize * sizeOfPlane;
    int sizeOfInputImg  = 0;

    DTYPE *delta_gpu       = this_delta->GetGPUData();
    DTYPE *input_delta_gpu = NULL;

    int preSize  = 0;

    for (int opnum = 0; opnum < m_noOperator; opnum++) {
        input_delta     = this->GetInput()[opnum]->GetDelta();
        input_delta_gpu = input_delta->GetGPUData();
        noBlock         = input_delta->GetChannelSize();
        preSize         = m_aAccumulate[opnum];
        sizeOfInputImg  = noBlock * sizeOfPlane;

        ConcatenateChannelWise_BackPropagate_kernel << < noBlock, threadsPerBlock >> > (sizeOfResultImg, sizeOfInputImg, timesize, batchsize, delta_gpu, input_delta_gpu, preSize);
    }

    // this->BackPropagate();

    return TRUE;
}

#endif  // ifdef __CUDNN__
