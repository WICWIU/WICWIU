#ifdef __CUDNN__

#include "Transpose.hpp"

template class Transpose<float>;

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


__global__ void Transpose_ForwardPropagate_kernel(float *result, float *input, int tiStride, int baStride, int chStride, int roStride, int coStride, int timesize, int batchsize, int channelsize, int rowsize, int colsize, int weightDim)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx<weightDim) {
        int resultShape[5] = {timesize, batchsize, channelsize, rowsize, colsize};
        int resultIdx[5] = {timesize, batchsize, channelsize, rowsize, colsize};

        GetTensorDimIndex(idx, resultIdx,weightDim);

        result[GetIdx(resultShape, resultIdx)] = input[tiStride*resultIdx[0]+baStride*resultIdx[1]+chStride*resultIdx[2]+roStride*resultIdx[3]+coStride*resultIdx[4]];
    }
}


template<typename DTYPE> int Transpose<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *result = this->GetResult();
    Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();

    Shape *resultTenShape = result->GetShape();
    Shape *inputTenShape = input->GetShape();

    DTYPE *result_gpu = result->GetGPUData();
    DTYPE *input_gpu  = input->GetGPUData();

    int timesize    = resultTenShape->GetDim(0);
    int batchsize   = resultTenShape->GetDim(1);
    int channelsize = resultTenShape->GetDim(2);
    int rowsize     = resultTenShape->GetDim(3);
    int colsize     = resultTenShape->GetDim(4);

    int* pStride;
    pStride = new int[5];

    for(int i = 4; i >= 0; i --){
        if(i == 4) pStride[i] = 1;
        else pStride[i] = pStride[i+1] * (inputTenShape->GetDim(i+1));
    }

    int temp = pStride[m_pDim0];
    pStride[m_pDim0] = pStride[m_pDim1];
    pStride[m_pDim1] = temp;

    int totalThread = this->GetResult()->GetCapacity();
    int noBlock, threadsPerBlock;
    GetKernelParameters(totalThread, &noBlock, &threadsPerBlock);

    Transpose_ForwardPropagate_kernel<< <noBlock, threadsPerBlock>> >(result_gpu, input_gpu, pStride[0], pStride[1], pStride[2], pStride[3], pStride[4], timesize, batchsize, channelsize, rowsize, colsize, totalThread);

    delete pStride;
    return true;
}


__global__ void Transpose_BackPropagate_kernel(float *input_delta, float *this_delta, int tiStride, int baStride, int chStride, int roStride, int coStride, int timesize, int batchsize, int channelsize, int rowsize, int colsize, int weightDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (idx < weightDim) {

        int inputDeltaShape[5] = {timesize, batchsize, channelsize, rowsize, colsize};
        int inputDeltaIdx[5] = {timesize, batchsize, channelsize, rowsize, colsize};

        GetTensorDimIndex(idx, inputDeltaIdx,weightDim);

        input_delta[GetIdx(inputDeltaShape, inputDeltaIdx)] = this_delta[inputDeltaIdx[0]*tiStride+inputDeltaIdx[1]*baStride+inputDeltaIdx[2]*chStride+inputDeltaIdx[3]*roStride+inputDeltaIdx[4]*coStride];
    }
}


template<typename DTYPE> int Transpose<DTYPE>::BackPropagateOnGPU(int pTime) {

    Tensor<DTYPE> *input_delta     = this->GetInput()[0]->GetDelta();
    Tensor<DTYPE> *this_delta      = this->GetDelta();

    Shape *pDeltaShape      = this_delta->GetShape();
    Shape *pInputDeltaShape = input_delta->GetShape();

    DTYPE *input_delta_gpu = input_delta->GetGPUData();
    DTYPE *this_delta_gpu  = this_delta->GetGPUData();

    int timesize    = pInputDeltaShape->GetDim(0);
    int batchsize   = pInputDeltaShape->GetDim(1);
    int channelsize = pInputDeltaShape->GetDim(2);
    int rowsize     = pInputDeltaShape->GetDim(3);
    int colsize     = pInputDeltaShape->GetDim(4);

    int* pStride;
    pStride = new int[5];

    for(int i = 4; i >= 0; i--){
        if (i == 4) pStride[i] = 1;
        else pStride[i] = pStride[i+1] * (pDeltaShape->GetDim(i+1));
    }

    int temp = pStride[m_pDim0];
    pStride[m_pDim0] = pStride[m_pDim1];
    pStride[m_pDim1] = temp;

    int totalThread = this->GetResult()->GetCapacity();
    int noBlock, threadsPerBlock;
    GetKernelParameters(totalThread, &noBlock, &threadsPerBlock);

    Transpose_BackPropagate_kernel<< <noBlock, threadsPerBlock>> >(input_delta_gpu, this_delta_gpu, pStride[0], pStride[1], pStride[2], pStride[3], pStride[4], timesize, batchsize, channelsize, rowsize, colsize, totalThread);

    return true;
}

template class TransposeTimeWise<float>;

__global__ void TransposeTimeWise_ForwardPropagate_kernel(float *result, float *input, int tiStride, int baStride, int chStride, int roStride, int coStride, int timesize, int batchsize, int channelsize, int rowsize, int colsize, int weightDim)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim; idx += blockDim.x * gridDim.x) {
        int resultShape[5] = {timesize, batchsize, channelsize, rowsize, colsize};
        int temp = batchsize*channelsize*rowsize*colsize;
        int cur = idx;

        for(int dim = 0; dim < 4; dim++) {
            resultShape[dim] = cur/temp;
            cur %= temp;
            temp /= resultShape[dim+1];
        }
        resultShape[4] = cur;

        result[idx] = input[tiStride*resultShape[0]+baStride*resultShape[1]+chStride*resultShape[2]+roStride*resultShape[3]+coStride*resultShape[4]];

    }

}

template<typename DTYPE> int TransposeTimeWise<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    int noBlock = 3, threadsPerBlock = 128;

    // std::cout<<"TransposeTimeWise Forward GPU"<<'\n';

    Tensor<DTYPE> *result = this->GetResult();
    Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();

    Shape *resultTenShape = result->GetShape();
    Shape *inputTenShape = input->GetShape();

    DTYPE *result_gpu = result->GetGPUData();
    DTYPE *input_gpu  = input->GetGPUData(pTime);

    int timesize    = resultTenShape->GetDim(0);
    int batchsize   = resultTenShape->GetDim(1);
    int channelsize = resultTenShape->GetDim(2);
    int rowsize     = resultTenShape->GetDim(3);
    int colsize     = resultTenShape->GetDim(4);

    int m_parameterDim = this->GetResult()->GetCapacity();       

    int* pStride;
    pStride = new int[5];

    for(int i=4; i>=0; i--){
        if(i==4) pStride[i] = 1;
        else pStride[i] = pStride[i+1]*(inputTenShape->GetDim(i+1));
    }

    int temp = pStride[m_pDim0];
    pStride[m_pDim0] = pStride[m_pDim1];
    pStride[m_pDim1] = temp;

    int inputTimeSize = input->GetTimeSize();
    int m_CapacityPerTime = input->GetCapacity() / inputTimeSize;
    DTYPE *x;
    checkCudaErrors(cudaMalloc((DTYPE**)&x, inputTimeSize * m_CapacityPerTime * sizeof(DTYPE)));

    for(int i=0; i<inputTimeSize; i++){

        DTYPE *wicwiuX       = input->GetGPUData(i);
        checkCudaErrors(cudaMemcpy(&x[m_CapacityPerTime*i], wicwiuX, (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
    }

    TransposeTimeWise_ForwardPropagate_kernel<< <noBlock, threadsPerBlock>> >(result_gpu, x, pStride[0], pStride[1], pStride[2], pStride[3], pStride[4], timesize, batchsize, channelsize, rowsize, colsize, m_parameterDim);

    checkCudaErrors(cudaFree(x));

    return true;
}


__global__ void TransposeTimeWise_BackPropagate_kernel(float *input_delta, float *this_delta, int tiStride, int baStride, int chStride, int roStride, int coStride, int timesize, int batchsize, int channelsize, int rowsize, int colsize, int weightDim)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim; idx += blockDim.x * gridDim.x) {
        int inputDeltaShape[5] = {timesize, batchsize, channelsize, rowsize, colsize};
        int temp = batchsize*channelsize*rowsize*colsize;
        int cur = idx;

        for(int dim = 0; dim < 4; dim++) {
            inputDeltaShape[dim] = cur/temp;
            cur %= temp;
            temp /= inputDeltaShape[dim+1];
        }
        inputDeltaShape[4] = cur;

        input_delta[idx] = this_delta[tiStride*inputDeltaShape[0]+baStride*inputDeltaShape[1]+chStride*inputDeltaShape[2]+roStride*inputDeltaShape[3]+coStride*inputDeltaShape[4]];
    }
}


template<typename DTYPE> int TransposeTimeWise<DTYPE>::BackPropagateOnGPU(int pTime) {
    int noBlock = 3, threadsPerBlock = 128;

    Tensor<DTYPE> *input_delta     = this->GetInput()[0]->GetDelta();
    Tensor<DTYPE> *this_delta      = this->GetDelta();

    Shape *pDeltaShape = this_delta->GetShape();
    Shape *pInputDeltaShape = input_delta->GetShape();

    DTYPE *input_delta_gpu = input_delta->GetGPUData(pTime);
    DTYPE *this_delta_gpu  = this_delta->GetGPUData();

    int timesize    = pInputDeltaShape->GetDim(0);
    int batchsize   = pInputDeltaShape->GetDim(1);
    int channelsize = pInputDeltaShape->GetDim(2);
    int rowsize     = pInputDeltaShape->GetDim(3);
    int colsize  = pInputDeltaShape->GetDim(4);

    int m_parameterDim = this->GetResult()->GetCapacity();

    int* pStride;
    pStride = new int[5];

    for(int i=4; i>=0; i--){
        if(i==4) pStride[i] = 1;
        else pStride[i] = pStride[i+1]*(pDeltaShape->GetDim(i+1));
    }
    int temp = pStride[m_pDim0];
    pStride[m_pDim0] = pStride[m_pDim1];
    pStride[m_pDim1] = temp;


    int inputTimeSize = input_delta->GetTimeSize();
    int m_CapacityPerTime = input_delta->GetCapacity() / inputTimeSize;

    TransposeTimeWise_BackPropagate_kernel<< <noBlock, threadsPerBlock>> >(x, this_delta_gpu, pStride[0], pStride[1], pStride[2], pStride[3], pStride[4],timesize, batchsize, channelsize, rowsize, colsize, m_parameterDim);


    for(int i=0; i<inputTimeSize; i++){

        DTYPE *wicwiuX       = input_delta->GetGPUData(i);
        checkCudaErrors(cudaMemcpy(wicwiuX, &x[m_CapacityPerTime*i], (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
    }

    return true;
}


#endif  // ifdef __CUDNN__
