#ifndef __LINEAR_LAYER__
#define __LINEAR_LAYER__    value

#include "../Module.hpp"

/*!
@class Linear Operator들을 그래프로 구성해 fully connected layer의 기능을 수행하는 모듈을 생성하는 클래스
@details Operator들을 뉴럴 네트워크의 서브 그래프로 구성해 fully connected layer의 기능을 수행한다
*/
template<typename DTYPE> class Linear : public Module<DTYPE>{
private:
public:
    /*!
    @brief Linear 클래스 생성자
    @details Linear 클래스의 Alloc 메소드를 호출한다.
    @see linear<DTYPE>::Alloc(Operator<DTYPE> *pInput, int pNumInputCol, int pNumOutputCol, int use_bias, std::string pName)
    */
    Linear(Operator<DTYPE> *pInput, int pNumInputCol, int pNumOutputCol, int use_bias = FALSE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, pNumInputCol, pNumOutputCol, use_bias, pName);
    }

    /*!
    @brief Linear 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.
    */
    virtual ~Linear() {}

    /*!
    @brief Linear(Fully Connected) Layer 그래프를 동적으로 할당 및 구성하는 메소드
    @details Input Operator의 Element에 대해 Weight를 이용해 행렬 곱(Matrix Multiplication)을 수행하고 Bias가 존재할 시 Bias를 합(Column Wise Addition)해 Output Operator로 내보내는 layer를 구성한다.
    @param pInput 해당 Layer의 Input에 해당하는 Operator
    @param pNumInputCol 해당 Layer의 Input Operator의 Column의 갯수, Input Column에 대한 Dimension
    @param pNumOutputCol 해당 Layer의 Output Operator의 Column의 갯수, Output Column에 대한 Dimension
    @param use_bias Bias 사용 유무, 0일 시 사용 안 함, 0이 아닐 시 사용
    @param pName Module의 이름
    @return TRUE
    @see MatMul<DTYPE>::MatMul(Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput, std::string pName) AddColWise<DTYPE>::AddColWise(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias, std::string pName) Module<DTYPE>::AnalyzeGraph(Operator<DTYPE> *pResultOperator)
    */
    int Alloc(Operator<DTYPE> *pInput, int pNumInputCol, int pNumOutputCol, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        // for He initialization
        float stddev = sqrt((float)4/(pNumInputCol + pNumOutputCol));
        // float stddev = 0.1;

        Tensorholder<DTYPE> *pWeight = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, pNumOutputCol, pNumInputCol, 0.0, stddev), "Layer_Weight_" + pName);
        out = new MatMul<DTYPE>(pWeight, out, "Layer_MatMul_" + pName);

        if (use_bias) {
            Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, pNumOutputCol, 0.f), "Add_Bias_" + pName);
            out = new AddColWise<DTYPE>(out, pBias, "Layer_Add_" + pName);
        }

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

#endif  // __LINEAR_LAYER__
