#ifndef __TRANSPOSEDCONVOLUTION_LAYER__
#define __TRANSPOSEDCONVOLUTION_LAYER__    value

#include "../Module.hpp"

/*!
@class TransposedConvolutionLayer2D Operator들을 그래프로 구성해 2-Dimensional TransposedConvolution Layer의 기능을 수행하는 모듈을 생성하는 클래스
@details Operator들을 뉴럴 네트워크의 서브 그래프로 구성해 2-Dimensional Transposedconvolution Layer의 기능을 수행한다
*/
template<typename DTYPE> class TransposedConvolutionLayer2D : public Module<DTYPE>{
private:
public:
    /*!
    @brief TransposedConvolutionLayer2D 클래스 생성자
    @details TransposedConvolutionLayer2D 클래스의 Alloc 함수를 호출한다.
    @see TransposedConvolutionLayer2D<DTYPE>::Alloc(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOutputChannel, int pNumKernelRow, int pNumKernelCol, int pStrideRow, int pStrideCol, int pPaddingRow, int pPaddingCol, int use_bias, std::string pName)
    */
    TransposedConvolutionLayer2D(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOutputChannel, int pNumKernelRow, int pNumKernelCol, int pStrideRow, int pStrideCol, int pPadding, int use_bias = FALSE, std::string pName = "NO NAME") : Module<DTYPE>(pName){
        Alloc(pInput, pNumInputChannel, pNumOutputChannel, pNumKernelRow, pNumKernelCol, pStrideRow, pStrideCol, pPadding, pPadding, use_bias, pName);
    }

    /*!
    @brief TransposedConvolutionLayer2D 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.
    */
    virtual ~TransposedConvolutionLayer2D() {}

    /*!
    @brief 2D TransposedConvolution Layer 그래프를 동적으로 할당 및 구성하는 메소드
    @details Input Operator의 Element에 대해 2D TransposedConvolution 수행한다.
    @details Input Operator의 Element에 대해 Weight를 이용해 2차원 전치합성 곱(2D TransposedConvolution)을 수행하고 Bias가 존재할 시 Bias를 합(Column Wise Addition)해 Output Operator로 내보내는 layer를 구성한다.
    @param pInput 해당 Layer의 Input에 해당하는 Operator
    @param pNumInputChannel 해당 Layer의 Input Operator의 Channel의 갯수, Input Column에 대한 Dimension
    @param pNumOutputChannel 해당 Layer의 Output Operator의 Channel의 갯수, Output Column에 대한 Dimension
    @param pNumKernelRow 2D TransposedConvolution Layer 커널의 Row Size
    @param pNumKernelCol 2D TransposedConvolution Layer 커널의 Column Size
    @param pStrideRow 2D TransposedConvolution Layer의 Row Stride Size
    @param pStrideCol 2D TransposedConvolution Layer의 Column Stride Size
    @param pPaddingRow 2D TransposedConvolution Layer의 Row Padding 값
    @param pPaddingCol 2D TransposedConvolution Layer의 Column Padding 값
    @param use_bias Bias 사용 유무, 0일 시 사용 안 함, 0이 아닐 시 사용
    @param pName Module의 이름
    @return TRUE
    @see TransposedConvolutionLayer2D<DTYPE>::TransposedConvolutionLayer2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int stride1, int stride2, std::string pName = "NO NAME") AddColWise<DTYPE>::AddColWise(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias, std::string pName) Module<DTYPE>::AnalyzeGraph(Operator<DTYPE> *pResultOperator)
    */
    int Alloc(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOutputChannel, int pNumKernelRow, int pNumKernelCol, int pStrideRow, int pStrideCol, int pPaddingRow, int pPaddingCol, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        Tensorholder<DTYPE> *pWeight = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, pNumInputChannel, pNumOutputChannel, pNumKernelRow, pNumKernelCol, 0.0, 0.1), "TransposedConvolution2D_Weight_" + pName);
        out = new TransposedConvolution2D<DTYPE>(out, pWeight, pStrideRow, pStrideCol, pPaddingRow, pPaddingCol, "TransposedConvolution2D_Convolution2D_" + pName);

        if(use_bias){
            Tensorholder<DTYPE> *pBias = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, pNumOutputChannel, 1, 1, 0), "TransposedConvolution2D_Bias_" + pName);
            out = new AddChannelWise<DTYPE>(out, pBias, "TransposedConvolution2D_Add_" + pName);
        }

        this->AnalyzeGraph(out);

        return TRUE;
    }
};


#endif  // __TRANSPOSEDCONVOLUTION_LAYER__
