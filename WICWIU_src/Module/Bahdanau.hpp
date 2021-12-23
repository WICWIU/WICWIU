#ifndef __BAHDANAUDECODER__
#define __BAHDANAUDECODER__    value

#include "../Module.hpp"
#include "Decoder.hpp"

/*!
 * @class RNN(LSTM or GRU)와 Attention Operator들을 그래프로 구성해 Bahdanau Attention Decoder의 기능을 수행하는 모듈을 생성하는 클래스
 * @details Operator들을 뉴럴 네트워크의 서브 그래프로 구성해 Bahdanau Attention Decoder의 기능을 수행한다
 * @details RNN, LSTM, GRU 중 선택하여 사용할 수 있다. 
*/
template<typename DTYPE> class Bahdanau : public Decoder<DTYPE>{
private:

    int timesize;    

    Operator<DTYPE> *m_initHiddenTensorholder;

    Operator<DTYPE> *m_encoderHidden;        

    Operator<DTYPE> *m_EncoderLengths;

    Operator<DTYPE> *m_Query;

public:

    /*!
     * @brief Bahdanau 클래스 생성자
     * @details Bahdanau 클래스의 Alloc 함수를 호출한다.
    */
    Bahdanau(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, Operator<DTYPE> *pMask, int vocabLength, int embeddingDim, int hiddensize, int outputsize, Operator<DTYPE> *pEncoderLengths = NULL, int use_bias = TRUE, std::string pName = "No Name") : Decoder<DTYPE>(pName) {
        Alloc(pInput, pEncoder, pMask, pEncoderLengths, vocabLength, embeddingDim, hiddensize, outputsize, use_bias, pName);
    }


    virtual ~Bahdanau() {}


    /**
     * @brief Bahdanau Attention Deocder그래프를 동적으로 할당 및 구성하는 메소드
     * @details RNN과 Attention을 사용하여 Bahdanau Attetnion Decoder 연산을 수행한다.
     * @param pInput 해당 Layer의 Input에 해당하는 Operator
     * @param pEncoder 해당 Bahdanau Attention에 연결될 Encoder Operator
     * @param pMask Attention연산 중 실제사용되지 않는 Tensor를 가려주는 Operator
     * @param pEncoderLengths Batch연산으로 인해 바뀌는 Encoder의 마지막 연산 결과의 time 위치를 저장하고 있는 Operator
     * @param vocabLength Bahdanau Attention에서 사용하는 전체 VOCAB의 개수
     * @param embeddingDim Bahdanau Attention에서 사용하는 embedding vector의 dimension
     * @param hiddensize Bahdanau Attention의 hidden size
     * @param outputsize Bahdanau Attention의 output dimenstion
     * @param use_bias RNN(LSTM or GRU) 연산에서 bias 사용 유무, 0일 시 사용 안함, 0이 아닐 시 사용
     * @param pName Module의 이름
     * @return int 
     */
    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, Operator<DTYPE> *pMask, Operator<DTYPE> *pEncoderLengths, int vocabLength, int embeddingDim, int hiddensize, int outputsize, int use_bias, std::string pName) {

        this->SetInput(3, pInput, pEncoder, pMask);           

        timesize = pInput->GetResult()->GetTimeSize();
        int batchsize = pInput->GetResult()->GetBatchSize();
        m_initHiddenTensorholder  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, batchsize, 1, 1, hiddensize), "tempHidden");

        m_EncoderLengths = pEncoderLengths;

        m_Query = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(timesize, batchsize, 1, 1, hiddensize), "m_Query_no");

        Operator<DTYPE> *out = pInput;

        //Embedding
        Operator<DTYPE> *embedding = new EmbeddingLayer<DTYPE>(out, vocabLength, embeddingDim, pName+"_Embedding");

        //Dropout
        embedding = new Dropout<DTYPE>(embedding, 0.1, "Dropout");

        Operator<DTYPE> *ContextVector = new BahdanauAttention<DTYPE>(pEncoder, m_Query, pEncoder, pMask, pName+"_BahdanauAttention");

        Operator<DTYPE> *concate = new ConcatenateColumnWise<DTYPE>(embedding,ContextVector, pName+"_concatenate");

        Operator<DTYPE> *hidden = new GRUCellLayer<DTYPE>(concate, embeddingDim+hiddensize*2, hiddensize, m_initHiddenTensorholder, use_bias, pName+"__Bahdanau__GRUCellLayer");   


        //linear
        out = new Linear<DTYPE>(hidden, hiddensize, outputsize, TRUE, pName+"_Fully-Connected-H2O");

        this->AnalyzeGraph(out);

        ContextVector->GetInputContainer()->Pop(m_Query);
        ContextVector->GetInputContainer()->Pop(pEncoder);
        ContextVector->GetInputContainer()->Pop(pMask);
        ContextVector->GetInputContainer()->Push(hidden);
        ContextVector->GetInputContainer()->Push(pEncoder);
        ContextVector->GetInputContainer()->Push(pMask);

        ContextVector->SetQuery(hidden);

        return TRUE;
    }

    /**
     * @brief Bahdanau Attention의 ForwradPropagate 메소드
     * @details Encoder의 마지막 time의 hidden값을 Decoder의 init hidden값으로 연결후 ForwradPropagate 연산을 수행한다.
     * @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
     * @return int 
     */
    int ForwardPropagate(int pTime=0) {

        Tensor<DTYPE> *_initHidden = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *initHidden = m_initHiddenTensorholder->GetResult();

        Shape *_initShape = _initHidden->GetShape();
        Shape *initShape = initHidden->GetShape();

        int enTimesize = _initHidden->GetTimeSize();
        int batchsize  = _initHidden->GetBatchSize();
        int colSize    = initHidden->GetColSize();                                    

        if( m_EncoderLengths != NULL){

            Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();

            for(int ba=0; ba<batchsize; ba++){
                for(int co=0; co<colSize; co++){
                    (*initHidden)[Index5D(initShape, 0, ba, 0, 0, co)] = (*_initHidden)[Index5D(_initShape, (*encoderLengths)[ba]-1, ba, 0, 0, co)];
                }
            }
        }
        else{
            for(int ba=0; ba<batchsize; ba++){
                for(int co=0; co<colSize; co++){
                    (*initHidden)[Index5D(initShape, 0, ba, 0, 0, co)] = (*_initHidden)[Index5D(_initShape, enTimesize-1, ba, 0, 0, co)];
                }
            }
        }

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = 0; i < numOfExcutableOperator; i++) {
            (*ExcutableOperator)[i]->ForwardPropagate(pTime);
        }


        return TRUE;
    }

    /**
     * @brief Bahdanau Attention의 BackPropagate 메소드
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

        Tensor<DTYPE> *enGradient = this->GetInput()[1]->GetGradient();
        Tensor<DTYPE> *_enGradient = m_initHiddenTensorholder->GetGradient();

        Shape *enShape  = enGradient->GetShape();
        Shape *_enShape = _enGradient->GetShape();

        int enTimesize = enGradient->GetTimeSize();
        int batchSize = enGradient->GetBatchSize();
        int colSize = _enGradient->GetColSize();                                     

        if( m_EncoderLengths != NULL){

            Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();

            for(int ba=0; ba < batchSize; ba++){
                for(int co=0; co < colSize; co++){
                    (*enGradient)[Index5D(enShape, (*encoderLengths)[ba]-1, ba, 0, 0, co)] += (*_enGradient)[Index5D(_enShape, 0, ba, 0, 0, co)];  
                }
            }

        }
        else{
            for(int ba=0; ba < batchSize; ba++){
                for(int co=0; co < colSize; co++){
                    (*enGradient)[Index5D(enShape, enTimesize-1, ba, 0, 0, co)] += (*_enGradient)[Index5D(_enShape, 0, ba, 0, 0, co)];
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


#endif  // __ATTENTIONDECODERMODULE__
