#ifdef __CUDNN__


#include "EmbeddingLayer.hpp"
#include "AttentionWeight.hpp"
#include "GRULayer.hpp"
#include "GRUCellLayer.hpp"
#include "LinearLayer.hpp"
#include "Attention.hpp"
#include "Bahdanau.hpp"



template class Bahdanau<float>;


__global__ void Bahdanau_ForwardPropagate_kernel(float *pDevEncoderHidden, float *pDevinitHidden, int batchIndex, int colSize) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < colSize; idx += blockDim.x * gridDim.x) {

          int startIndex = colSize * batchIndex;

          pDevinitHidden[startIndex + idx] = pDevEncoderHidden[startIndex + idx];

    }
}

template<typename DTYPE> int Bahdanau<DTYPE>::ForwardPropagateOnGPU(int pTime) {


      if(pTime == 0){

          int noBlock = 3, threadsPerBlock = 128;
          Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();
          Tensor<DTYPE> *_initHidden = this->GetInput()[1]->GetResult();
          Tensor<DTYPE> *initHidden = m_initHiddenTensorholder->GetResult();
          int batchsize  = _initHidden->GetBatchSize();
          int colSize    = initHidden->GetColSize();

          for(int ba = 0; ba < batchsize; ba++){
              DTYPE *m_pDevEncoderHidden  = _initHidden->GetGPUData((*encoderLengths)[ba]-1);        
              DTYPE *m_pDevinitHidden  = initHidden->GetGPUData(0);

              Bahdanau_ForwardPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevEncoderHidden, m_pDevinitHidden, ba, colSize);
          }
      }

      int numOfExcutableOperator = this->GetNumOfExcutableOperator();
      Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

      for (int i = 0; i < numOfExcutableOperator; i++) {
          (*ExcutableOperator)[i]->ForwardPropagateOnGPU(pTime);
      }
      return TRUE;
}


__global__ void Bahdanau_BackPropagate_kernel(float *pDevEncoderHidden, float *pDevDecoderHidden, int batchIndex, int colSize) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < colSize; idx += blockDim.x * gridDim.x) {
          int startIndex = colSize * batchIndex;        
          pDevEncoderHidden[startIndex*2 + idx] += pDevDecoderHidden[startIndex + idx];
    }
}

template<typename DTYPE> int Bahdanau<DTYPE>::BackPropagateOnGPU(int pTime) {

      if(pTime !=0)
        return TRUE;

      int numOfExcutableOperator = this->GetNumOfExcutableOperator();
      Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

      for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
          for(int ti = timesize-1; ti>=0; ti--){
            (*ExcutableOperator)[i]->BackPropagateOnGPU(ti);
        }
      }

      if(pTime == 0){
          int noBlock = 3, threadsPerBlock = 128;

          Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();
          Tensor<DTYPE> *_initHidden = this->GetInput()[1]->GetGradient();
          Tensor<DTYPE> *initHidden = m_initHiddenTensorholder->GetGradient();
          int batchsize  = _initHidden->GetBatchSize();
          int colSize    = initHidden->GetColSize();

          for(int ba = 0; ba < batchsize; ba++){
              DTYPE *m_pDevEncoderHidden  = _initHidden->GetGPUData((*encoderLengths)[ba]-1);
              DTYPE *m_pDevinitHidden  = initHidden->GetGPUData(0);
              Bahdanau_BackPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevEncoderHidden, m_pDevinitHidden, ba, colSize);
          }
      }
      return TRUE;
}

#endif  // ifdef __CUDNN__
