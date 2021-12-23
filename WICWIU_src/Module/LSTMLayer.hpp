#ifndef __LSTM_LAYER__
#define __LSTM_LAYER__    value

#include "../Module.hpp"

/*!
@class LSTM Operator들을 그래프로 구성해 LSTM Layer의 기능을 수행하는 모듈을 생성하는 클래스
@details Operator들을 뉴럴 네트워크의 서브 그래프로 구성해 LSTM Layer의 기능을 수행한다
*/
template<typename DTYPE> class LSTMLayer : public Module<DTYPE>{
private:
public:
    /*!
     * @brief LSTMLayer 클래스 생성자
     * @details LSTMLayer 클래스의 Alloc 함수를 호출한다.
    */
    LSTMLayer(Operator<DTYPE> *pInput, int inputSize, int hiddenSize,  Operator<DTYPE> *initHidden, int useBias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputSize, hiddenSize, initHidden, useBias, pName);
    }

    /*!
     * @brief LSTMLayer 클래스 소멸자
     * @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.  
     */
    virtual ~LSTMLayer() {}

    /*!
     * @brief LSTMLayer 그래프를 동적으로 할당 및 구성하는 메소드
     * @details Input Operator의 Element에 대해 LSTM연산을 수행한다.
     * @param pInput\해당 Layer의 Input에 해당하는 Operator
     * @param inputSize 해당 Layer의 Input Operator의 Column의 갯수
     * @param hiddenSize LSTM의 hidden size
     * @param initHidden LSTM의 init hidden에 해당하는 Operator
     * @param useBias Bias 사용 유무, 0일 시 사용 안 함, 0이 아닐 시 사용
     * @param pName Module의 이름
     * @return TRUE
    */
    int Alloc(Operator<DTYPE> *pInput, int inputSize, int hiddenSize, Operator<DTYPE> *initHidden, int useBias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        //weight
        Tensorholder<DTYPE> *pWeightIG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 4*hiddenSize, inputSize, 0.0, 0.01), "LSTMLayer_pWeight_IG_" + pName);
        Tensorholder<DTYPE> *pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 4*hiddenSize, hiddenSize, 0.0, 0.01), "LSTMLayer_pWeight_HG_" + pName);

        //bias
        Tensorholder<DTYPE> *lstmBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, 4*hiddenSize, 0.f), "RNN_Bias_f" + pName);


#ifdef __CUDNN__
        pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddenSize, 4*hiddenSize+4*inputSize+4, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);    //bias 1개 일때!!!        //이거 output으로 바꿔서 해보기!!!
#endif  // __CUDNN__

        out = new LSTM<DTYPE>(out, pWeightIG, pWeightHG, lstmBias, initHidden);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};


#endif  // __LSMT_LAYER__
