#ifndef __BIGRU_LAYER__
#define __BIGRU_LAYER__    value

#include "../Module.hpp"



/*!
@class GRU Operator들을 그래프로 구성해 Bidirectional GRU Layer의 기능을 수행하는 모듈을 생성하는 클래스
@details Operator들을 뉴럴 네트워크의 서브 그래프로 구성해 Bidirectional GRU Layer의 기능을 수행한다
*/
template<typename DTYPE> class BidirectionalGRULayer : public Module<DTYPE>{
private:

      Operator<DTYPE> * reversedInput;

public:

    /*!
     * @brief BidirectionalGRULayer 클래스 생성자
     * @details BidirectionalGRULayer 클래스의 Alloc 함수를 호출한다.
    */
    BidirectionalGRULayer(Operator<DTYPE> *pInput, int inputsize, int hiddensize, Operator<DTYPE> *initHidden, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputsize, hiddensize, initHidden, use_bias, pName);
    }
    virtual ~BidirectionalGRULayer() {}

    /**
     * @brief BidirectionalGRULayer 그래프를 동적으로 할당 및 구성하는 메소드
     * @details Input Operator의 Element에 대해 LSTM연산을 수행한다.
     * @param pInput\해당 Layer의 Input에 해당하는 Operator
     * @param inputSize 해당 Layer의 Input Operator의 Column의 갯수
     * @param hiddenSize GRU의 hidden size
     * @param initHidden GRU의 init hidden에 해당하는 Operator
     * @param useBias Bias 사용 유무, 0일 시 사용 안 함, 0이 아닐 시 사용
     * @param pName Module의 이름
     * @return TRUE
    */
    int Alloc(Operator<DTYPE> *pInput, int inputsize, int hiddensize, Operator<DTYPE> *initHidden, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        reversedInput = new FlipTimeWise<DTYPE>(pInput, "pInput_FLIP");

        //weight & bias for forward
        Tensorholder<DTYPE> *pWeightIG_f = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddensize, inputsize, 0.0, 0.01), "BiGRULayer_pWeight_IG_f_" + pName);
        Tensorholder<DTYPE> *pWeightHG_f = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddensize, hiddensize, 0.0, 0.01), "BiGRULayer_pWeight_HG_f_" + pName);
        Tensorholder<DTYPE> *pWeightICH_f = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "BiGRULayer_pWeight_ICH_f_" + pName);
        Tensorholder<DTYPE> *pWeightHCH_f = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "BiGRULayer_pWeight_HCH_f_" + pName);

        Tensorholder<DTYPE> *gBias_f = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, 2*hiddensize, 0.f), "BiGRU_Bias_g_f_" + pName);
        Tensorholder<DTYPE> *chBias_f = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "BiGRU_Bias_ch_f_" + pName);

        //weight & bias for backward
        Tensorholder<DTYPE> *pWeightIG_b = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddensize, inputsize, 0.0, 0.01), "BiGRULayer_pWeight_IG_b_" + pName);
        Tensorholder<DTYPE> *pWeightHG_b = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 2*hiddensize, hiddensize, 0.0, 0.01), "BiGRULayer_pWeight_HG_b_" + pName);
        Tensorholder<DTYPE> *pWeightICH_b = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "BiGRULayer_pWeight_ICH_b_" + pName);
        Tensorholder<DTYPE> *pWeightHCH_b = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "BiGRULayer_pWeight_HCH_b_" + pName);

        Tensorholder<DTYPE> *gBias_b = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, 2*hiddensize, 0.f), "BiGRU_Bias_g_b_" + pName);
        Tensorholder<DTYPE> *chBias_b = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "BiGRU_Bias_ch_b_" + pName);



#ifdef __CUDNN__
        pWeightHG_f = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, (3*hiddensize+3*inputsize+3), 0.0, 0.01), "__f__BiDiGRULayer_pWeight_h2h_" + pName);
        pWeightHG_b = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, (3*hiddensize+3*inputsize+3), 0.0, 0.01), "__b__BiDiGRULayer_pWeight_h2h_" + pName);
        Operator<DTYPE> *Fout = new GRU<DTYPE>(out, pWeightIG_f, pWeightHG_f, pWeightICH_f, pWeightHCH_f, gBias_f, chBias_f, initHidden);
        Operator<DTYPE> *Bout = new GRU<DTYPE>(reversedInput, pWeightIG_b, pWeightHG_b, pWeightICH_b, pWeightHCH_b, gBias_b, chBias_b, initHidden);
        Bout = new FlipTimeWise<DTYPE>(Bout, "FLIP_AFTER_GRU");                 
        Operator<DTYPE> * concate = new ConcatenateColumnWise<DTYPE>(Fout,Bout, "Bidirectional_concatenate");
#else

        Operator<DTYPE> *Fout = new GRU<DTYPE>(out, pWeightIG_f, pWeightHG_f, pWeightICH_f, pWeightHCH_f, gBias_f, chBias_f, initHidden);
        Operator<DTYPE> *Bout = new GRU<DTYPE>(reversedInput, pWeightIG_b, pWeightHG_b, pWeightICH_b, pWeightHCH_b, gBias_b, chBias_b, initHidden);
        Bout = new FlipTimeWise<DTYPE>(Bout, "FLIP_AFTER_BiGRU");               

        Operator<DTYPE> * concate = new ConcatenateColumnWise<DTYPE>(Fout,Bout, "Bidirectional_concatenate");

#endif  // __CUDNN__

        this->AnalyzeGraph(concate);

        return TRUE;
    }

    int ForwardPropagate(int pTime=0) {

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        int timesize = (*this->GetInputContainer())[0]->GetResult()->GetTimeSize();

        if(pTime != timesize-1) return TRUE;

        for(int i = 0; i< numOfExcutableOperator; i++){
            for(int pTime = 0; pTime < timesize; pTime++){
                (*ExcutableOperator)[i]->ForwardPropagate(pTime);
            }
        }

        return TRUE;
    }

    int BackPropagate(int pTime=0) {

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        int timesize = (*this->GetInputContainer())[0]->GetResult()->GetTimeSize();

        if(pTime != timesize-1) return TRUE;

        for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
            for(int pTime = timesize -1; pTime >=0 ; pTime--){
              (*ExcutableOperator)[i]->BackPropagate(pTime);
            }
        }
        return TRUE;
    }

#ifdef __CUDNN__

    int ForwardPropagateOnGPU(int pTime=0) {

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        int timesize = (*this->GetInputContainer())[0]->GetResult()->GetTimeSize();

        if(pTime != timesize-1) return TRUE;

        for(int i = 0; i< numOfExcutableOperator; i++){
            for(int pTime = 0; pTime < timesize; pTime++){
                (*ExcutableOperator)[i]->ForwardPropagateOnGPU(pTime);
            }
        }

        return TRUE;
    }

    int BackPropagateOnGPU(int pTime=0) {

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        int timesize = (*this->GetInputContainer())[0]->GetResult()->GetTimeSize();

        if(pTime != timesize-1) return TRUE;

        for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
            for(int pTime = timesize -1; pTime >=0 ; pTime--){
            (*ExcutableOperator)[i]->BackPropagateOnGPU(pTime);
            }
        }
        return TRUE;
    }

#endif

};

#endif
