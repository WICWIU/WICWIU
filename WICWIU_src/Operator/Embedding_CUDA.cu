#ifdef __CUDNN__

#include "Embedding.hpp"

template class Embedding<float>;

__global__ void ForwardPropagate_kernel(float *pDevInput, float *pDevWeight, float *pDevOutput, int batchsize, int iputcolsize, int numOfWord, int embeddingDim) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numOfWord; idx += blockDim.x * gridDim.x) {

          //printf("idx : %d\n", idx);

          for (int ba = 0; ba < batchsize; ba++) {


               int wordIndex = pDevInput[ba*iputcolsize + idx];
               //printf("wordIndex : %d\n", wordIndex);

               for(int co =0; co < embeddingDim; co++){
                  pDevOutput[ba*embeddingDim*numOfWord + idx*embeddingDim + co] = pDevWeight[wordIndex*embeddingDim + co];
               }


          }

    }
}

template<typename DTYPE> int Embedding<DTYPE>::ForwardPropagateOnGPU(int pTime) {
        int noBlock = 3, threadsPerBlock = 128;


        // std::cout<<"Embedding<DTYPE>::ForwardPropagateOnGPU : "<<pTime<<'\n';

        Tensor<DTYPE> *weight  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        // std::cout<<weight->GetShape()<<'\n';
        // std::cout<<input->GetShape()<<'\n';

        //2 : INFERENCE  SentenceTranslate 할때 잘 동작하나 확인하기 위한 코드
        // if(this->GetMode() == 2){
        //     std::cout<<"Embedding<DTYPE>::ForwardPropagateOnGPU : "<<pTime<<'\n';
        //     std::cout<<input->GetShape()<<'\n';
        //     std::cout<<input<<'\n';
        // }

        int batchsize        = result->GetBatchSize();
        int inputcolsize     = input->GetColSize();
        int numOfWord        = result->GetRowSize();
        int embeddingDim     = result->GetColSize();

        DTYPE *m_pDevInput  = input->GetGPUData(pTime);
        DTYPE *m_pDevWeight  = weight->GetGPUData();
        DTYPE *m_pDevOutput = result->GetGPUData(pTime);

        // std::cout<<"kernel 함수 호출 전"<<'\n';
        // std::cout<<numOfWord<<'\n';

        ForwardPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevInput, m_pDevWeight, m_pDevOutput, batchsize, inputcolsize, numOfWord, embeddingDim);

        return TRUE;
}


__global__ void BackPropagate_kernel(float *pDevInput, float *pDevWeight, float *pDevOutput, int batchsize, int iputcolsize, int numOfWord, int embeddingDim) {

      for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numOfWord; idx += blockDim.x * gridDim.x) {

            //printf("idx : %d\n", idx);

            for (int ba = 0; ba < batchsize; ba++) {

                 int wordIndex = pDevInput[ba*iputcolsize + idx];
                 // printf("wordIndex : %d\n", wordIndex);

                 for(int co =0; co < embeddingDim; co++){
                    pDevWeight[wordIndex*embeddingDim + co] += pDevOutput[ba*embeddingDim*numOfWord + idx*embeddingDim + co];
                 }


            }

      }

}


template<typename DTYPE> int Embedding<DTYPE>::BackPropagateOnGPU(int pTime) {
        int noBlock = 3, threadsPerBlock = 128;

        // std::cout<<"Embedding<DTYPE>::BackPropagateOnGPU : "<<pTime<<'\n';

        Tensor<DTYPE> *weight  = this->GetInput()[0]->GetGradient();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetGradient();

        //위에서 잘 전달해줌!
        // if(pTime == 0){
        //   std::cout<<result->GetShape()<<'\n';    //이제는 위에 layer에서 모든 time에 대해서 호출하고 나서 호출하니깐 다 들어가있는게 맞지
        //   std::cout<<result<<'\n';
        // }

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
