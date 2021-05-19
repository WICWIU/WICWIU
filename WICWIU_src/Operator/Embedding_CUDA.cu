#ifdef __CUDNN__

#include "Embedding.hpp"

template class Embedding<float>;

__global__ void ForwardPropagate_kernel(float *pDevInput, float *pDevWeight, float *pDevOutput, int batchsize, int iputcolsize, int numOfWord, int embeddingDim) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numOfWord; idx += blockDim.x * gridDim.x) {

          for (int ba = 0; ba < batchsize; ba++) {
               int wordIndex = pDevInput[ba*iputcolsize + idx];

               for(int co =0; co < embeddingDim; co++){
                  pDevOutput[ba*embeddingDim*numOfWord + idx*embeddingDim + co] = pDevWeight[wordIndex*embeddingDim + co];
               }
          }
    }
}

template<typename DTYPE> int Embedding<DTYPE>::ForwardPropagateOnGPU(int pTime) {
        int noBlock = 3, threadsPerBlock = 128;

        Tensor<DTYPE> *weight  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();


        int batchsize        = result->GetBatchSize();
        int inputcolsize     = input->GetColSize();
        int numOfWord        = result->GetRowSize();
        int embeddingDim     = result->GetColSize();

        DTYPE *m_pDevInput  = input->GetGPUData(pTime);
        DTYPE *m_pDevWeight  = weight->GetGPUData();
        DTYPE *m_pDevOutput = result->GetGPUData(pTime);

        ForwardPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevInput, m_pDevWeight, m_pDevOutput, batchsize, inputcolsize, numOfWord, embeddingDim);

        return TRUE;
}


__global__ void BackPropagate_kernel(float *pDevInput, float *pDevWeight, float *pDevOutput, int batchsize, int iputcolsize, int numOfWord, int embeddingDim) {

      for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numOfWord; idx += blockDim.x * gridDim.x) {
            for (int ba = 0; ba < batchsize; ba++) {
                 int wordIndex = pDevInput[ba*iputcolsize + idx];
                 for(int co =0; co < embeddingDim; co++){
                    pDevWeight[wordIndex*embeddingDim + co] += pDevOutput[ba*embeddingDim*numOfWord + idx*embeddingDim + co];
                 }
            }
      }
}


template<typename DTYPE> int Embedding<DTYPE>::BackPropagateOnGPU(int pTime) {
        int noBlock = 3, threadsPerBlock = 128;

        Tensor<DTYPE> *weight  = this->GetInput()[0]->GetGradient();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetGradient();

        int batchsize        = result->GetBatchSize();
        int inputcolsize     = input->GetColSize();
        int numOfWord        = result->GetRowSize();
        int embeddingDim     = result->GetColSize();

        DTYPE *m_pDevInput  = input->GetGPUData(pTime);
        DTYPE *m_pDevWeight  = weight->GetGPUData();
        DTYPE *m_pDevOutput = result->GetGPUData(pTime);

        BackPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevInput, m_pDevWeight, m_pDevOutput, batchsize, inputcolsize, numOfWord, embeddingDim);

        return TRUE;
}

#endif  // ifdef __CUDNN__
