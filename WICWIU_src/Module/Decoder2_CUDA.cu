#ifdef __CUDNN__



// #include "../LossFunction/SoftmaxCrossEntropy.hpp"
//
// // template class SoftmaxCrossEntropy<int>;
// template class SoftmaxCrossEntropy<float>;

// #include "Decoder2.hpp"
#include "LinearLayer.hpp"
#include "EmbeddingLayer.hpp"
#include "RecurrentLayer.hpp"
#include "LSTM2Layer.hpp"
#include "GRULayer.hpp"
#include "Decoder2.hpp"


// template class LRelu<int>;
template class Decoder2<float>;

/*!
@class LRelu cuda
*/
__global__ void ForwardPropagate_kernel(float *pDevEncoderHidden, float *pDevinitHidden, int batchIndex, int colSize) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < colSize; idx += blockDim.x * gridDim.x) {

          int startIndex = colSize * batchIndex;

          pDevinitHidden[startIndex + idx] = pDevEncoderHidden[startIndex + idx];

    }
}
/*!
  EncoderLengths가 NULL일때는 고려하지 않고 구현!!!
  추후 추가 필요!
*/

template<typename DTYPE> int Decoder2<DTYPE>::ForwardPropagateOnGPU(int pTime) {

      // std::cout<<"Decoder2<DTYPE>::ForwardPropagateOnGPU "<<pTime<<'\n';

  //encoder에서 decoder로 복사!
      if(pTime == 0){

          int noBlock = 3, threadsPerBlock = 128;

          Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();

          //Data 접근!
          Tensor<DTYPE> *_initHidden = this->GetInput()[1]->GetResult();
          Tensor<DTYPE> *initHidden = m_initHiddenTensorholder->GetResult();

          // std::cout<<"Encoder last hidden value"<<'\n';
          // std::cout<<_initHidden->GetShape()<<'\n';
          // std::cout<<_initHidden<<'\n';

          //batchsize, colsize
          int batchsize  = _initHidden->GetBatchSize();
          int colSize    = _initHidden->GetColSize();

          // std::cout<<"복사 전"<<'\n';
          // std::cout<<initHidden->GetShape()<<'\n';
          // std::cout<<initHidden<<'\n';

          // std::cout<<"batch size : "<<batchsize<<'\n';

          for(int ba = 0; ba < batchsize; ba++){

              DTYPE *m_pDevEncoderHidden  = _initHidden->GetGPUData((*encoderLengths)[ba]-1);
              DTYPE *m_pDevinitHidden  = initHidden->GetGPUData(0);

              ForwardPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevEncoderHidden, m_pDevinitHidden, ba, colSize);

          }

          // std::cout<<"복사 해온 값"<<'\n';
          // std::cout<<initHidden->GetShape()<<'\n';
          // std::cout<<initHidden<<'\n';
          // //
          // std::cout<<"Encoder length"<<'\n';
          // std::cout<<encoderLengths<<'\n';

      }

      int numOfExcutableOperator = this->GetNumOfExcutableOperator();
      Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

      for (int i = 0; i < numOfExcutableOperator; i++) {
          (*ExcutableOperator)[i]->ForwardPropagateOnGPU(pTime);
      }
      return TRUE;
}

/*!
@brief LRelu의 BackPropagate 커널함수
@details BackPropagateOnGPU에서 호출되어 실행
@see int LRelu<DTYPE>::BackPropagateOnGPU(int pTime = 0)
@details 1차원으로 배열 된 block과 thread에 접근하여 연산
@param pDevOutput LRelu ForwardPropagate연산의 결과인 output값의 GPU data
@param pDevDelta LRelu 다음 Operator의 BackPropagate 결과 값인 delta의 GPU data.
@param pDevInputDelta 연산의 결과인 delta값을 저장할 GPU data.
@param negativeSlope output값이 0.f 이하일 때 사용하는 기울기값
@param weightDim LRelu연산의 결과값의 dimension.
*/
__global__ void BackPropagate_kernel(float *pDevEncoderHidden, float *pDevDecoderHidden, int batchIndex, int colSize) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < colSize; idx += blockDim.x * gridDim.x) {

          int startIndex = colSize * batchIndex;

          pDevEncoderHidden[startIndex + idx] = pDevDecoderHidden[startIndex + idx];

    }
}


template<typename DTYPE> int Decoder2<DTYPE>::BackPropagateOnGPU(int pTime) {

      //seq2seq에 맞춰서 수정하기!!! --> Decoder안에 embedding, rnn, linear 다 있어서 layer별로 time을 다 실행하고 나서 넘어가야 됨!
      if(pTime !=0)
        return TRUE;
      //그래서 딱 한번만 호출되고 안쪽에서 모든 time에 대해 처리하도록!

      int numOfExcutableOperator = this->GetNumOfExcutableOperator();
      Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

      // std::cout<<"backward 호출 전 initHidden gradient 값"<<'\n';
      // std::cout<<m_initHiddenTensorholder->GetGradient()<<'\n';

      // for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
      //     (*ExcutableOperator)[i]->BackPropagateOnGPU(pTime);                   // 여기 어차피 한번만 돌려버리면 되는거 아닌가  --> embedding도 있어서 안됨!
      // }

      //seq2seq에 맞춰서 수정하기!!! --> Decoder안에 embedding, rnn, linear 다 있어서 layer별로 time을 다 실행하고 나서 넘어가야 됨!
      for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
          for(int ti = timesize-1; ti>=0; ti--){
            (*ExcutableOperator)[i]->BackPropagateOnGPU(ti);
        }
      }


      //decoder에서 encoder로 복사!
      if(pTime == 0){

          int noBlock = 3, threadsPerBlock = 128;

          Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();

          //Data 접근!
          Tensor<DTYPE> *_initHidden = this->GetInput()[1]->GetGradient();
          Tensor<DTYPE> *initHidden = m_initHiddenTensorholder->GetGradient();

          //batchsize, colsize
          int batchsize  = _initHidden->GetBatchSize();
          int colSize    = _initHidden->GetColSize();

          // std::cout<<"복사 전 encoder의 gradient"<<'\n';
          // std::cout<<_initHidden<<'\n';

          // std::cout<<"initHidden의 gradient"<<'\n';
          // std::cout<<initHidden<<'\n';

          for(int ba = 0; ba < batchsize; ba++){

              DTYPE *m_pDevEncoderHidden  = _initHidden->GetGPUData((*encoderLengths)[ba]-1);
              DTYPE *m_pDevinitHidden  = initHidden->GetGPUData(0);

              BackPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevEncoderHidden, m_pDevinitHidden, ba, colSize);

          }

          // std::cout<<"복사 후 encoder의 gradient"<<'\n';
          // std::cout<<_initHidden<<'\n';
          //
          //
          // std::cout<<"Encoder length"<<'\n';
          // std::cout<<encoderLengths<<'\n';

      }

      return TRUE;
}

#endif  // ifdef __CUDNN__
