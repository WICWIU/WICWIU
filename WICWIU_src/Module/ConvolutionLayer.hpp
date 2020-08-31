#ifndef __CONVOLUTION_LAYER__
#define __CONVOLUTION_LAYER__ value

#include "../Module.hpp"

// template<typename DTYPE> class ConvolutionLayer2D : public Module<DTYPE>{
// private:
// public:
//     ConvolutionLayer2D(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOutputChannel, int
//     pNumKernelRow, int pNumKernelCol, int pStrideRow, int pStrideCol, int pPadding, int use_bias
//     = FALSE, std::string pName = "NO NAME") : Module<DTYPE>(pName){
//         Alloc(pInput, pNumInputChannel, pNumOutputChannel, pNumKernelRow, pNumKernelCol,
//         pStrideRow, pStrideCol, pPadding, pPadding, use_bias, pName);
//     }
//
//     virtual ~ConvolutionLayer2D() {}
//
//     int Alloc(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOutputChannel, int
//     pNumKernelRow, int pNumKernelCol, int pStrideRow, int pStrideCol, int pPaddingRow, int
//     pPaddingCol, int use_bias, std::string pName) {
//         Operator<DTYPE> *out = pInput;
//
//         Tensorholder<DTYPE> *pWeight = (Tensorholder<DTYPE> *)this->AddParameter(new
//         Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, pNumOutputChannel, pNumInputChannel,
//         pNumKernelRow, pNumKernelCol, 0.0, 0.1), "Convolution2D_Weight_" + pName)); out =
//         this->AddOperator(new Convolution2D<DTYPE>(out, pWeight, pStrideRow, pStrideCol,
//         pPaddingRow, pPaddingCol, "Convolution2D_Convolution2D_" + pName));
//
//         if(use_bias){
//             Tensorholder<DTYPE> *pBias = (Tensorholder<DTYPE> *)this->AddParameter(new
//             Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, pNumOutputChannel, 1, 1, 0),
//             "Convolution2D_Bias_" + pName)); out = this->AddOperator(new
//             AddChannelWise<DTYPE>(out, pBias, "Convolution2D_Add_" + pName));
//         }
//
//         return TRUE;
//     }
// };

/*!
@class ConvolutionLayer2D Operator들을 그래프로 구성해 2-Dimensional Convolution Layer의 기능을
수행하는 모듈을 생성하는 클래스
@details Operator들을 뉴럴 네트워크의 서브 그래프로 구성해 2-Dimensional convolution Layer의 기능을
수행한다
*/
template <typename DTYPE>
class ConvolutionLayer2D : public Module<DTYPE>
{
private:
public:
    /*!
    @brief ConvolutionLayer2D 클래스 생성자
    @details ConvouutionLayer2D 클래스의 Alloc 함수를 호출한다.
    @see Convolution2D<DTYPE>::Alloc(Operator<DTYPE> *pInput, int pNumInputChannel, int
    pNumOutputChannel, int pNumKernelRow, int pNumKernelCol, int pStrideRow, int pStrideCol, int
    pPaddingRow, int pPaddingCol, int use_bias, std::string pName)
    */
    ConvolutionLayer2D(Operator<DTYPE>* pInput, int pNumInputChannel, int pNumOutputChannel,
                       int pNumKernelRow, int pNumKernelCol, int pStrideRow, int pStrideCol,
                       int pPadding, int use_bias = FALSE, std::string pName = "NO NAME")
        : Module<DTYPE>(pName)
    {
        Alloc(pInput, pNumInputChannel, pNumOutputChannel, pNumKernelRow, pNumKernelCol, pStrideRow,
              pStrideCol, pPadding, pPadding, use_bias, pName);
    }

    /*!
    @brief ConvolutionLayer2D 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.
    */
    virtual ~ConvolutionLayer2D() {}

    /*!
    @brief 2D convolution Layer 그래프를 동적으로 할당 및 구성하는 메소드
    @details Input Operator의 Element에 대해 2D Convolution을 수행한다.
    @details Input Operator의 Element에 대해 Weight를 이용해 2차원 합성 곱(2D Convolution)을
    수행하고 Bias가 존재할 시 Bias를 합(Column Wise Addition)해 Output Operator로 내보내는 layer를
    구성한다.
    @param pInput 해당 Layer의 Input에 해당하는 Operator
    @param pNumInputChannel 해당 Layer의 Input Operator의 Channel의 갯수, Input Channel에 대한
    Dimension
    @param pNumOutputChannel 해당 Layer의 Output Operator의 Channel의 갯수, Output Channel에 대한
    Dimension
    @param pNumKernelRow 2D Convolution Layer 커널의 Row Size
    @param pNumKernelCol 2D Convolution Layer 커널의 Column Size
    @param pStrideRow 2D Convolution Layer의 Row Stride Size
    @param pStrideCol 2D Convolution Layer의 Column Stride Size
    @param pPaddingRow 2D Convolution Layer의 Row Padding 값
    @param pPaddingCol 2D Convolution Layer의 Column Padding 값
    @param use_bias Bias 사용 유무, 0일 시 사용 안 함, 0이 아닐 시 사용
    @param pName Module의 이름
    @return TRUE
    @see Convolution2D<DTYPE>::Convolution2D(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, int
    stride1, int stride2, std::string pName = "NO NAME")
    AddColWise<DTYPE>::AddColWise(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias, std::string
    pName) Module<DTYPE>::AnalyzeGraph(Operator<DTYPE> *pResultOperator)
    */
    int Alloc(Operator<DTYPE>* pInput, int pNumInputChannel, int pNumOutputChannel,
              int pNumKernelRow, int pNumKernelCol, int pStrideRow, int pStrideCol, int pPaddingRow,
              int pPaddingCol, int use_bias, std::string pName)
    {
        this->SetInput(pInput);

        Operator<DTYPE>* out = pInput;

        // for He initialization
        // int n_i = pNumKernelRow * pNumKernelCol * pNumInputChannel;
        // int n_o = pNumKernelRow * pNumKernelCol * pNumOutputChannel;
        // float stddev = sqrt((float)4/(n_i + n_o));
        // std::cout << stddev << '\n';
        // float stddev = 0.1;

        // Tensorholder<DTYPE> *pWeight = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1,
        // pNumOutputChannel, pNumInputChannel, pNumKernelRow, pNumKernelCol, 0.0, stddev),
        // "Convolution2D_Weight_" + pName);
        Tensorholder<DTYPE>* pWeight = new Tensorholder<DTYPE>(
            Tensor<DTYPE>::Random_normal(1, pNumOutputChannel, pNumInputChannel, pNumKernelRow,
                                         pNumKernelCol, 0.0, 0.02),
            "Convolution2D_Weight_" + pName);
        out = new Convolution2D<DTYPE>(out, pWeight, pStrideRow, pStrideCol, pPaddingRow,
                                       pPaddingCol, "Convolution2D_Convolution2D_" + pName);

        if (use_bias)
        {
            Tensorholder<DTYPE>* pBias =
                new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, pNumOutputChannel, 1, 1, 0),
                                        "Convolution2D_Bias_" + pName);
            out = new AddChannelWise<DTYPE>(out, pBias, "Convolution2D_Add_" + pName);
        }

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

#endif // __CONVOLUTION_LAYER__
