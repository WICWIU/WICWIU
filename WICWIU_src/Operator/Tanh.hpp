#ifndef TANH_H_
#define TANH_H_ value

#include "../Operator.hpp"

template <typename DTYPE>
class Tanh : public Operator<DTYPE>
{
public:
    /*!
    @brief Tanh의 생성자
    @details 파라미터로 받은 pInput으로 Alloc한다.
    @param pInput Alloc할 대상 Operator
    @param pName Operator에 사용자가 부여한 이름.
    */
    Tanh(Operator<DTYPE>* pInput, std::string pName, int pLoadflag = TRUE)
        : Operator<DTYPE>(pInput, pName, pLoadflag)
    {
#ifdef __DEBUG__
        std::cout << "Tanh::Tanh(Operator *)" << '\n';
#endif // __DEBUG__
        this->Alloc(pInput);
    }

    /*!
    @brief Tanh의 소멸자.
    */
    ~Tanh() { std::cout << "Tanh::~Tanh()" << '\n'; }

    /*!
    @brief 파라미터로 받은 pInput으로부터 맴버 변수들을 초기화 한다.
    @details Result와 Gradient를 저장하기 위해 pInput의 Shape과 같은 dim을 갖는 Tensor를 생성한다.
    @param pInput 생성 할 Tensor의 Shape정보를 가진 Operator
    @return 성공 시 TRUE.
    */
    int Alloc(Operator<DTYPE>* pInput)
    {
#ifdef __DEBUG__
        std::cout << "Tanh::Alloc(Operator *, Operator *)" << '\n';
#endif // __DEBUG__

        int timesize = pInput->GetResult()->GetTimeSize();
        int batchsize = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize = pInput->GetResult()->GetRowSize();
        int colsize = pInput->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    /*!
    @brief Tanh의 ForwardPropagate 매소드
    @details input의 Tensor값들을 Tanh을 취한 뒤 result에 저장한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int ForwardPropagate(int pTime = 0)
    {
        Tensor<DTYPE>* input = this->GetInput()[0]->GetResult();
        Tensor<DTYPE>* result = this->GetResult();

        int timesize = result->GetTimeSize();
        int batchsize = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize = result->GetRowSize();
        int colsize = result->GetColSize();

        Shape* resultTenShape = result->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++)
        {
            for (int ch = 0; ch < channelsize; ch++)
            {
                for (int ro = 0; ro < rowsize; ro++)
                {
                    for (int co = 0; co < colsize; co++)
                    {
                        (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)] =
                            tanh((*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)]);
                    }
                }
            }
        }

        return TRUE;
    }

    /*!
    @brief Tanh의 BackPropagate 매소드.
    @details result값으로 tanh의 미분 값을 계산하여 input_delta에 더한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0)
    {
        Tensor<DTYPE>* result = this->GetResult();
        Tensor<DTYPE>* this_delta = this->GetDelta();
        Tensor<DTYPE>* input_delta = this->GetInput()[0]->GetDelta();

        int timesize = result->GetTimeSize();
        int batchsize = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize = result->GetRowSize();
        int colsize = result->GetColSize();

        Shape* resultTenShape = result->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++)
        {
            for (int ch = 0; ch < channelsize; ch++)
            {
                for (int ro = 0; ro < rowsize; ro++)
                {
                    for (int co = 0; co < colsize; co++)
                    {
                        (*input_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)] +=
                            (1 - (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]) *
                            (1 + (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]) *
                            (*this_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)];
                    }
                }
            }
        }

        return TRUE;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime)
    {
        this->ForwardPropagate(pTime);
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime)
    {
        this->BackPropagate(pTime);

        return TRUE;
    }

#endif // __CUDNN__
};

#endif // TANH_H_
