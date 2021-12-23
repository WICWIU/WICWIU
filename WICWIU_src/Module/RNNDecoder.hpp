#ifndef __RNN_DECODER__
#define __RNN_DECODER__    value

#include "../Module.hpp"
#include "Decoder.hpp"

/*!
 * @class RNN(LSTM or GRU) Operator들을 그래프로 구성해 Decoder의 기능을 수행하는 모듈을 생성하는 클래스
 * @details Operator들을 뉴럴 네트워크의 서브 그래프로 구성해 Decoder의 기능을 수행한다
 * @details RNN, LSTM, GRU 중 선택하여 사용할 수 있다.
*/
template<typename DTYPE> class RNNDecoder : public Decoder<DTYPE>{
private:

    int timesize;

    Operator<DTYPE> *m_initHiddenTensorholder;

    Operator<DTYPE> *m_EncoderLengths;

    int m_isTeacherForcing;

public:

    /*!
     * @brief RNNDecoder 클래스 생성자
     * @details RNNDecoder 클래스의 Alloc 함수를 호출한다.
    */
    RNNDecoder(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, int vocabLength, int embeddingDim, int hiddenSize, int outputSize, int m_isTeacherForcing = TRUE, Operator<DTYPE> *pEncoderLengths = NULL, int useBias = TRUE, std::string pName = "No Name") : Decoder<DTYPE>(pName) {
        Alloc(pInput, pEncoder, vocabLength, embeddingDim, hiddenSize, outputSize, m_isTeacherForcing, pEncoderLengths, useBias, pName);
    }


    virtual ~RNNDecoder() {}


    /**
     * @brief RNNDecoder 그래프를 동적으로 할당 및 구성하는 메소드
     * @details Input Operator의 Element를 embedding을 통해 vector로 변환 후 RNN 연산을 수행한다.
     * @details Encoder의 결과값을 받아 Decoder의 init hidden으로 사용한다.
     * @param pInput 해당 Layer의 Input에 해당하는 Operator
     * @param pEncoder 해당 Decoder에 연결될 Encoder Operator
     * @param vocabLength RNNDecoder에서 사용하는 전체 VOCAB의 개수
     * @param embeddingDim RNNDecoder에서 사용하는 embedding vector의 dimension
     * @param hiddenSize RNNDecoder의 hidden size
     * @param outputSize RNNDecoder의 output dimenstion
     * @param teacherForcing TeacherFrocing 사용 유무
     * @param pEncoderLengths Batch연산으로 인해 바뀌는 Encoder의 마지막 연산 결과의 time 위치를 저장하고 있는 Operator
     * @param useBias RNN(LSTM or GRU) 연산에서 bias 사용 유무, 0일 시 사용 안함, 0이 아닐 시 사용
     * @param pName Module의 이름
     * @return int 
     */
    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, int vocabLength, int embeddingDim, int hiddenSize, int outputSize, int teacherForcing, Operator<DTYPE> *pEncoderLengths, int useBias, std::string pName) {

        this->SetInput(2, pInput, pEncoder);

        timesize = pInput->GetResult()->GetTimeSize();
        int batchsize = pInput->GetResult()->GetBatchSize();
        m_initHiddenTensorholder  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, batchsize, 1, 1, hiddenSize), "tempHidden");

        m_EncoderLengths = pEncoderLengths;
        m_isTeacherForcing = teacherForcing;

        Operator<DTYPE> *out = pInput;

        out = new EmbeddingLayer<DTYPE>(out, vocabLength, embeddingDim, "Embedding");

         // out = new RecurrentLayer<DTYPE>(out, embeddingDim, hiddenSize, m_initHiddenTensorholder, useBias, "Recur_1");
        // out = new LSTMLayer<DTYPE>(out, embeddingDim, hiddenSize, m_initHiddenTensorholder, useBias, "Recur_1");
        out = new GRULayer<DTYPE>(out, embeddingDim, hiddenSize, m_initHiddenTensorholder, useBias, "Recur_1");


        out = new Linear<DTYPE>(out, hiddenSize, outputSize, useBias, "Fully-Connected-H2O");

        this->AnalyzeGraph(out);

        return TRUE;
    }

    /**
     * @brief RNNDecoder의 ForwradPropagate 메소드
     * @details Encoder의 마지막 time의 hidden값을 Decoder의 init hidden값으로 연결후 ForwradPropagate 연산을 수행한다.
     * @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
     * @return int 
     */
    int ForwardPropagate(int pTime=0) {

        if(pTime == 0){
              Tensor<DTYPE> *_initHidden = this->GetInput()[1]->GetResult();
              Tensor<DTYPE> *initHidden = m_initHiddenTensorholder->GetResult();

              Shape *_initShape = _initHidden->GetShape();
              Shape *initShape = initHidden->GetShape();

              int enTimesize = _initHidden->GetTimeSize();
              int batchsize  = _initHidden->GetBatchSize();
              int colSize    = _initHidden->GetColSize();

              if( m_EncoderLengths != NULL){

                  Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();

                  for(int ba=0; ba<batchsize; ba++){
                      for(int co=0; co<colSize; co++){
                          (*initHidden)[Index5D(initShape, 0, ba, 0, 0, co)] = (*_initHidden)[Index5D(_initShape, (*encoderLengths)[ba]-1, ba, 0, 0, co)];
                      }
                  }
              }else{

                  for(int ba=0; ba<batchsize; ba++){
                      for(int co=0; co<colSize; co++){
                          (*initHidden)[Index5D(initShape, 0, ba, 0, 0, co)] = (*_initHidden)[Index5D(_initShape, enTimesize-1, ba, 0, 0, co)];
                      }
                  }
            }
        }

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = 0; i < numOfExcutableOperator; i++)
            (*ExcutableOperator)[i]->ForwardPropagate(pTime);

        return TRUE;
    }

    /**
     * @brief RNNDecoder의 BackPropagate 메소드
     * @details Encoder의 마지막 time의 hidden값을 Decoder의 init hidden값으로 연결후 BackPropagate 연산을 수행한다.
     * @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
     * @return int 
     */
    int BackPropagate(int pTime=0) {

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
            (*ExcutableOperator)[i]->BackPropagate(pTime);
        }

        if(pTime == 0){
              Tensor<DTYPE> *enGradient = this->GetInput()[1]->GetGradient();
              Tensor<DTYPE> *_enGradient = m_initHiddenTensorholder->GetGradient();

              Shape *enShape  = enGradient->GetShape();
              Shape *_enShape = _enGradient->GetShape();

              int enTimesize = enGradient->GetTimeSize();
              int batchSize = enGradient->GetBatchSize();
              int colSize = enGradient->GetColSize();


              if( m_EncoderLengths != NULL){

                  Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();

                  for(int ba=0; ba < batchSize; ba++){
                      for(int co=0; co < colSize; co++){
                          (*enGradient)[Index5D(enShape, (*encoderLengths)[ba]-1, ba, 0, 0, co)] = (*_enGradient)[Index5D(_enShape, 0, ba, 0, 0, co)];
                      }
                  }

              }
              else{
                  for(int ba=0; ba < batchSize; ba++){
                      for(int co=0; co < colSize; co++){
                          (*enGradient)[Index5D(enShape, enTimesize-1, ba, 0, 0, co)] = (*_enGradient)[Index5D(_enShape, 0, ba, 0, 0, co)];
                      }
                  }
              }

        }

        return TRUE;
    }


  #ifdef __CUDNN__
      int ForwardPropagateOnGPU(int pTime = 0);
      int BackPropagateOnGPU(int pTime = 0);
  #endif // CUDNN

};

#endif  // __RNN_DECODER__
