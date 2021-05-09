#ifndef __RECURRENT_LAYER__
#define __RECURRENT_LAYER__    value

#include "../Module.hpp"


template<typename DTYPE> class RecurrentLayer : public Module<DTYPE>{
private:
public:
    /*!
    @brief RecurrentLayer 클래스 생성자
    @details RecurrentLayer 클래스의 Alloc 함수를 호출한다.*/
    RecurrentLayer(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int outputsize, Operator<DTYPE> *initHidden, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, inputsize, hiddensize, outputsize, initHidden, use_bias, pName);
    }

    /*!
    @brief Recurrentlayer 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.  */
    virtual ~RecurrentLayer() {}

    /*!
    @brief RecurrentLayer 그래프를 동적으로 할당 및 구성하는 메소드
    @details Input Operator의 Element에 대해 2D Convolution을 수행한다.
    @param pInput
    @param use_bias Bias 사용 유무, 0일 시 사용 안 함, 0이 아닐 시 사용
    @param pName Module의 이름
    @return TRUE
    @see
    */
    int Alloc(Operator<DTYPE> *pInput, int inputsize, int hiddensize, int outputsize, Operator<DTYPE>* initHidden, int use_bias, std::string pName) {
        this->SetInput(pInput);
        // this->SetInput(2, pInput, initHidden);      //이렇게 하면 문제가 생김...

        Operator<DTYPE> *out = pInput;

        //--------------------------------------------초기화 방법-------------------------
        float xavier_i = sqrt(2/(inputsize+hiddensize));
        float xavier_h = sqrt(2/(hiddensize+hiddensize));
        float xavier_o = sqrt(2/(inputsize+outputsize));

        //원래는 0.01로 해둠
        Tensorholder<DTYPE> *pWeight_x2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, inputsize, 0.0, 0.01), "RecurrentLayer_pWeight_x2h_" + pName);
        Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);
        //Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::IdentityMatrix(1, 1, 1, hiddensize, hiddensize), "RecurrentLayer_pWeight_h2h_" + pName);

        //cudnn때문에 추가
#ifdef __CUDNN__
        //Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize+inputsize, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);
        //Tensorholder<DTYPE> *pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize+inputsize, hiddensize, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);
        pWeight_h2h = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, hiddensize, hiddensize+inputsize+1, 0.0, 0.01), "RecurrentLayer_pWeight_h2h_" + pName);    //bias 1개 일때!!!        //이거 output으로 바꿔서 해보기!!!
#endif  // __CUDNN__


        //Tensorholder<DTYPE> *pWeight_h2o = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, outputsize, hiddensize, 0.0, 0.01), "RecurrentLayer_pWeight_h2o_" + pName);

        //recurrent 내에 bias 추가 하는 거!
        Tensorholder<DTYPE> *rBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, hiddensize, 0.f), "RNN_Bias_" + pName);

        //out = new SeqRecurrent<DTYPE>(out, pWeight_x2h, pWeight_h2h, rBias);
        //out = new RecurrentCUDNN2<DTYPE>(out, pWeight_x2h, pWeight_h2h, rBias);                   //gpu 사용할때는 이걸!!!

        //initHidden값이 NULL이면 내부에서 알아서 처리해주지!!!
        out = new SeqRecurrent<DTYPE>(out, pWeight_x2h, pWeight_h2h, rBias, initHidden);


        //이제 h2o 부분은 밖으로 제외시킴!!!
        // out = new MatMul<DTYPE>(pWeight_h2o, out, "rnn_matmul_ho");
        //
        // if (use_bias) {
        //     Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, outputsize, 0.f), "Add_Bias_" + pName);
        //     out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        // }

        this->AnalyzeGraph(out);

        return TRUE;
    }
//
// #ifdef __CUDNN__
// int ForwardPropagate(int pTime=0) {
//
//     int numOfExcutableOperator = this->GetNumOfExcutableOperator();
//     Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();
//
//     for(int ti=0; ti<timesize; ti++){
//         for (int i = 0; i < numOfExcutableOperator; i++) {
//             (*ExcutableOperator)[i]->ForwardPropagate(ti);
//         }
//     }
//
//
//     return TRUE;
// }
//
// int BackPropagate(int pTime=0) {
//
//     //if(pTime)
//
//     int numOfExcutableOperator = this->GetNumOfExcutableOperator();
//     Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();
//
//     for(int ti=timesize-1; ti>=0; ti--){
//         for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
//             (*ExcutableOperator)[i]->BackPropagate(ti);
//         }
//     }
//
//
//
//     return TRUE;
// }
//
// #endif  // __CUDNN__

};


#endif  // __RECURRENT_LAYER__
