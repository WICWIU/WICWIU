#ifndef __RNN_ENCODER__
#define __RNN_ENCODER__    value

#include "../Module.hpp"

/*!
 * @class RNN(LSTM or GRU) Operator들을 그래프로 구성해 Encoder의 기능을 수행하는 모듈을 생성하는 클래스
 * @details Operator들을 뉴럴 네트워크의 서브 그래프로 구성해 Encoder의 기능을 수행한다
 * @details RNN, LSTM, GRU 중 선택하여 사용할 수 있다.
*/
template<typename DTYPE> class RNNEncoder : public Encoder<DTYPE>{
private:

    int timesize;

public:
    /*!
     * @brief RNNEncoder 클래스 생성자
     * @details RNNEncoder 클래스의 Alloc 함수를 호출한다.
    */
    RNNEncoder(Operator<DTYPE> *pInput, int vocabLength, int embeddingDim, int hiddenSize, int useBias = TRUE, std::string pName = "No Name") : Encoder<DTYPE>(pName) {
        Alloc(pInput, vocabLength, embeddingDim, hiddenSize, useBias, pName);
    }


    virtual ~RNNEncoder() {}

    /**
     * @brief RNNEncoder 그래프를 동적으로 할당 및 구성하는 메소드
     * @details Input Operator의 Element를 embedding을 통해 vector로 변환 후 RNN 연산을 수행한다.
     * @param pInput 해당 Layer의 Input에 해당하는 Operator
     * @param vocabLength RNNEncoder에서 사용하는 전체 VOCAB의 개수
     * @param embeddingDim RNNEncoder에서 사용하는 embedding vector의 dimension
     * @param hiddenSize RNNEncoder의 hidden size
     * @param useBias RNN(LSTM or GRU) 연산에서 bias 사용 유무, 0일 시 사용 안함, 0이 아닐 시 사용
     * @param pName Module의 이름
     * @return int 
     */
    int Alloc(Operator<DTYPE> *pInput, int vocabLength, int embeddingDim, int hiddenSize, int useBias, std::string pName) {

        timesize = pInput->GetResult()->GetTimeSize();
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        out = new EmbeddingLayer<DTYPE>(out, vocabLength, embeddingDim, "Embedding");

        // out = new RecurrentLayer<DTYPE>(out, embeddingDim, hiddenSize, NULL, useBias, "Recur_1");
        // out = new LSTMLayer<DTYPE>(out, embeddingDim, hiddenSize, NULL, useBias, "Recur_1");
        out = new GRULayer<DTYPE>(out, embeddingDim, hiddenSize, NULL, useBias, "Recur_1");
        // out = new BidirectionalGRULayer<DTYPE>(out, embeddingDim, hiddenSize, NULL, useBias, pName+"_GRU");



        this->AnalyzeGraph(out);

        return TRUE;
    }

    int ForwardPropagate(int pTime=0) {


        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = 0; i < numOfExcutableOperator; i++) {
            (*ExcutableOperator)[i]->ForwardPropagate(pTime);
        }

        return TRUE;
    }

    int BackPropagate(int pTime=0) {


        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
            (*ExcutableOperator)[i]->BackPropagate(pTime);
        }



        return TRUE;
    }
};



#endif  // __RNN_ENCODER__
