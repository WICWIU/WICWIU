#ifndef __LSTM2_LAYER__
#define __LSTM2_LAYER__    value

#include "../Module.hpp"


template<typename DTYPE> class LSTM2Layer : public Module<DTYPE>{
private:
public:
    /*!
    @brief LSTMLayer 클래스 생성자
    @details LSTMLayer 클래스의 Alloc 함수를 호출한다.*/
    LSTM2Layer(Operator<DTYPE> *pInput, int inputsize, int hiddensize,  Operator<DTYPE> *initHidden, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputsize, hiddensize, initHidden, use_bias, pName);
    }

    /*!
    @brief LSTMLayer 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.  */
    virtual ~LSTM2Layer() {}

    /*!
    @brief LSTMLayer 그래프를 동적으로 할당 및 구성하는 메소드
    @details Input Operator의 Element에 대해 2D Convolution을 수행한다.
    @param pInput
    @param use_bias Bias 사용 유무, 0일 시 사용 안 함, 0이 아닐 시 사용
    @param pName Module의 이름
    @return TRUE
    @see
    */
    int Alloc(Operator<DTYPE> *pInput, int inputsize, int hiddensize, Operator<DTYPE> *initHidden, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;


        //weight 2개
        Tensorholder<DTYPE> *pWeightIG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 4*hiddensize, inputsize, 0.0, 0.01), "LSTMLayer_pWeight_IG_" + pName);
        Tensorholder<DTYPE> *pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, 4*hiddensize, hiddensize, 0.0, 0.01), "LSTMLayer_pWeight_HG_" + pName);

        //output으로 나가는 weight
        // Tensorholder<DTYPE> *pWeight_h2o = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, outputsize, hiddensize, 0.0, 0.01), "LSTMLayer_pWeight_HO_" + pName);

        //bias 1개
        Tensorholder<DTYPE> *lstmBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, 4*hiddensize, 0.f), "RNN_Bias_f" + pName);


#ifdef __CUDNN__
        pWeightHG = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, 4*hiddensize+4*inputsize+4, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);    //bias 1개 일때!!!        //이거 output으로 바꿔서 해보기!!!
#endif  // __CUDNN__

        //out = new LSTM2<DTYPE>(out, pWeightIG, pWeightHG, lstmBias);

        out = new SeqLSTM2<DTYPE>(out, pWeightIG, pWeightHG, lstmBias, initHidden);


        //이제 h2o 부분 밖으로 뺴기!!!
        // out = new MatMul<DTYPE>(pWeight_h2o, out, "rnn_matmul_ho");
        //
        // if (use_bias) {
        //     Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, outputsize, 0.f), "Add_Bias_" + pName);
        //     out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        // }

        this->AnalyzeGraph(out);

        return TRUE;
    }
};


#endif  // __LSMT2_LAYER__
