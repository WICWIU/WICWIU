#ifdef __CUDNN__

#include "SoftmaxCrossEntropy.hpp"

// template class SoftmaxCrossEntropy<int>;
template class SoftmaxCrossEntropy<float>;
// template class SoftmaxCrossEntropy<double>;

__global__ void SoftmaxCrossEntropy_ForwardPropagate_kernel(int time, int batchsize, int colsize, float epsilon, float *result, float *label, float *softmaxresult) {
    int result_idx = 0;
    int start      = 0;
    int end        = 0;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batchsize; idx += blockDim.x * gridDim.x) {
        result_idx = time * batchsize + idx;
        start      = result_idx * colsize;
        end        = start + colsize;

        for (int i = start; i < end; i++) {
            result[result_idx] += -label[i] * log(softmaxresult[i] + epsilon);
            // result[result_idx] += -label[i] * log(MAX(softmaxresult[i], softmaxresult[i] + epsilon));
        }
    }
}

template<typename DTYPE> Tensor<DTYPE> *SoftmaxCrossEntropy<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *input         = this->GetTensor();
    Tensor<DTYPE> *label         = this->GetLabel()->GetResult();
    Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;
    Tensor<DTYPE> *result        = this->GetResult();

    int batchsize = input->GetBatchSize();
    int colsize   = input->GetColSize();

    float alpha = 1.f;
    float beta  = 0.f;

    cudnnTensorDescriptor_t pInputDesc   = input->GetDescriptor();
    cudnnTensorDescriptor_t pSoftMaxDesc = softmaxresult->GetDescriptor();

    DTYPE *pDevInput   = input->GetGPUData(pTime);
    DTYPE *pDevSoftMax = softmaxresult->GetGPUData(pTime);

    checkCUDNN(cudnnSoftmaxForward(this->GetCudnnHandle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                   &alpha, pInputDesc, pDevInput,
                                   &beta, pSoftMaxDesc, pDevSoftMax));

    int noBlock = 3, threadsPerBlock = 128;
    GetKernelParameters(batchsize, &noBlock, &threadsPerBlock);

    DTYPE *pDevLabel  = label->GetGPUData(pTime);
    DTYPE *pDevResult = result->GetGPUData(pTime);

    SoftmaxCrossEntropy_ForwardPropagate_kernel << < noBlock, threadsPerBlock >> > (0, batchsize, colsize, m_epsilon, pDevResult, pDevLabel, pDevSoftMax);

    return result;
}

__global__ void SoftmaxCrossEntropy_BackPropagate_kernel(int time, int capacity, float *input_delta, float *label, float *softmaxresult) {
    int idx = 0;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < capacity; idx += blockDim.x * gridDim.x) {
        idx = time * capacity + idx;

        input_delta[idx] = softmaxresult[idx] - label[idx];
    }
}

__global__ void SoftmaxCrossEntropy_BackPropagate_kernel_padding(int time, int batchIndex, int capacity, float *input_delta, float *label, float *softmaxresult) {
    int idx = 0;


    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < capacity; idx += blockDim.x * gridDim.x) {
        idx = batchIndex * capacity + idx;

        input_delta[idx] = softmaxresult[idx] - label[idx];
    }
}

template<typename DTYPE> Tensor<DTYPE> *SoftmaxCrossEntropy<DTYPE>::BackPropagateOnGPU(int pTime) {
    Tensor<DTYPE> *label         = this->GetLabel()->GetResult();
    Tensor<DTYPE> *softmaxresult = m_aSoftmaxResult;
    Tensor<DTYPE> *input_delta   = this->GetOperator()->GetDelta();

    int batchsize = input_delta->GetBatchSize();
    int colsize   = input_delta->GetColSize();
    int capacity  = batchsize * colsize;

    DTYPE *pDevSoftMax    = softmaxresult->GetGPUData(pTime);
    DTYPE *pDevLabel      = label->GetGPUData(pTime);
    DTYPE *pDevInputDelta = input_delta->GetGPUData(pTime);

    int noBlock = 3, threadsPerBlock = 128;
    GetKernelParameters(capacity, &noBlock, &threadsPerBlock);

    if(m_PaddingLengths != NULL){
        capacity = colsize;
        GetKernelParameters(capacity, &noBlock, &threadsPerBlock);
        Tensor<DTYPE> *Lengths = m_PaddingLengths->GetResult();
        for(int ba = 0; ba < batchsize; ba++){
            if((*Lengths)[ba] <= pTime)  continue;
            SoftmaxCrossEntropy_BackPropagate_kernel_padding << < noBlock, threadsPerBlock >> > (0, ba, capacity, pDevInputDelta, pDevLabel, pDevSoftMax);
        }

    }else{
      SoftmaxCrossEntropy_BackPropagate_kernel << < noBlock, threadsPerBlock >> > (0, capacity, pDevInputDelta, pDevLabel, pDevSoftMax);
    }

    return NULL;
}

#endif  // ifdef __CUDNN__
