#ifndef __GRU_LAYER__
#define __GRU_LAYER__    value

#include "../Module.hpp"

/*!
@class GRU Operator들을 그래프로 구성해 GRU Layer의 기능을 수행하는 모듈을 생성하는 클래스
@details Operator들을 뉴럴 네트워크의 서브 그래프로 구성해 GRU Layer의 기능을 수행한다
*/
template<typename DTYPE> class GRULayer : public Module<DTYPE>{
private:
public:

    /*!
     * @brief GRULayer 클래스 생성자
     * @details GRULayer 클래스의 Alloc 함수를 호출한다.
    */
    GRULayer(Operator<DTYPE> *pInput, int inputSize, int hiddenSize, Operator<DTYPE> *initHidden, int useBias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputSize, hiddenSize, initHidden, useBias, pName);
    }
    virtual ~GRULayer() {}

     /**
     * @brief GRULayer 그래프를 동적으로 할당 및 구성하는 메소드
     * @details Input Operator의 Element에 대해 LSTM연산을 수행한다.
     * @param pInput\해당 Layer의 Input에 해당하는 Operator
     * @param inputSize 해당 Layer의 Input Operator의 Column의 갯수
     * @param hiddenSize GRU의 hidden size
     * @param initHidden GRU의 init hidden에 해당하는 Operator
     * @param useBias Bias 사용 유무, 0일 시 사용 안 함, 0이 아닐 시 사용
     * @param pName Module의 이름
     * @return TRUE
    */
    int Alloc(Operator<DTYPE> *pInput, int inputSize, int hiddenSize, Operator<DTYPE> *initHidden, int useBias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        //weight
        Tensorholder<DTYPE> *pWeightIG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddenSize, inputSize, 0.0, 0.01), "GRULayer_pWeight_IG_" + pName);
        Tensorholder<DTYPE> *pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddenSize, hiddenSize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);
        Tensorholder<DTYPE> *pWeightICH = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddenSize, inputSize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);
        Tensorholder<DTYPE> *pWeightHCH = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddenSize, hiddenSize, 0.0, 0.01), "GRULayer_pWeight_HG_" + pName);

        //bias
        Tensorholder<DTYPE> *gBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, 2*hiddenSize, 0.f), "RNN_Bias_f" + pName);
        Tensorholder<DTYPE> *chBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddenSize, 0.f), "RNN_Bias_f" + pName);

#ifdef __CUDNN__
        pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddenSize, 3*hiddenSize+3*inputSize+3, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);    //For 1 bias option
#endif  // __CUDNN__

        out = new GRU<DTYPE>(out, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias, initHidden);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

#endif
