#ifndef MSE_H_
#define MSE_H_ value

#include "../LossFunction.hpp"

/*!
@class MSE MSE(Mean Squared Error) Metric를 이용해 뉴럴 네트워크의 손실 함수를 계산하는 클래스
@details MSE(Mean Squared Error) 계산 식을 이용해 뉴럴 네트워크의 순전파를 통해 계산된 출력 Tensor와
레이블 값의 손실 함수를 계산한다
*/
template <typename DTYPE>
class MSE : public LossFunction<DTYPE>
{
public:
    /*!
    @brief MSE(Mean Squared Error) LossFunction 클래스 생성자
    @details LossFunction 클래스의 생성자를 호출하고, Operator를 매개변수로 전달하여
    MSE<DTYPE>::Alloc(Operator<DTYPE> *pOperator) 메소드를 호출한다.
    @param pOperator MSE<DTYPE>::Alloc(Operator<DTYPE> *pOperator) 메소드의 매개변수로 전달할
    Operator
    @param pLabel LossFunction의 입력 레이블에 해당하는 Operator
    @param pName LossFunction의 이름
    @return 없음
    @see MSE<DTYPE>::Alloc(Operator<DTYPE> *pOperator)
    */
    MSE(Operator<DTYPE>* pOperator, Operator<DTYPE>* pLabel, std::string pName)
        : LossFunction<DTYPE>(pOperator, pLabel, pName)
    {
#ifdef __DEBUG__
        std::cout << "MSE::MSE(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
#endif // __DEBUG__
        this->Alloc(pOperator);
    }

    /*!
    @brief MSE(Mean Squared Error) LossFunction 클래스 소멸자
    @return 없음
    */
    virtual ~MSE()
    {
#ifdef __DEBUG__
        std::cout << "MSE::~MSE()" << '\n';
#endif // __DEBUG__
    }

    /*!
    @brief MSE(Mean Squared Error) LossFunction의 멤버 변수들을 동적 할당하는 메소드
    @details 매개변수로 전달받은 Operator를 Input Operator에 할당하고 초기화 된 Result 텐서를
    동적으로 할당 및 생성한다.
    @param pOperator MSE LossFunction의 입력에 해당하는 Operator
    @return TRUE
    */
    virtual int Alloc(Operator<DTYPE>* pOperator)
    {
#ifdef __DEBUG__
        std::cout << "MSE::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
#endif // __DEBUG__

        Operator<DTYPE>* pInput = pOperator;

        int timesize = pInput->GetResult()->GetTimeSize();
        int batchsize = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize = pInput->GetResult()->GetRowSize();
        int colsize = pInput->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        return TRUE;
    }

    /*!
    @brief MSE(Mean Squared Error) LossFunction의 순전파를 수행하는 메소드
    @details 구성한 뉴럴 네트워크에서 얻어진 결과 값을 레이블 값과 비교해 Mse(Mean Squared Error)를
    구한다
    @param pTime 입력 Tensor의 Time 축의 Dimension
    @return 뉴럴 네트워크의 결과 값에 대한 MSE(Mean Squared Error)
    */
    Tensor<DTYPE>* ForwardPropagate(int pTime = 0)
    {
        Tensor<DTYPE>* input = this->GetTensor();
        Tensor<DTYPE>* label = this->GetLabel()->GetResult();
        Tensor<DTYPE>* result = this->GetResult();

        int batchsize = input->GetBatchSize();

        int channelsize = input->GetChannelSize();
        int rowsize = input->GetRowSize();
        int colsize = input->GetColSize();
        int capacity = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int ba = 0, i = 0; ba < batchsize; ba++)
        {
            i = ti * batchsize + ba;

            for (int j = 0, index = 0; j < capacity; j++)
            {
                index = i * capacity + j;
                (*result)[i] += Error((*input)[index], (*label)[index]);
            }
        }

        return result;
    }

    /*!
    @brief MSE(Mean Squared Error) LossFunction의 역전파를 수행하는 메소드
    @details 구성한 뉴럴 네트워크에서 얻어진 MSE(Mean Squared Error)에 대한 입력 Tensor의 Gradient를
    계산한다
    @param pTime 입력 Tensor의 Time 축의 Dimension
    @return NULL
    */
    Tensor<DTYPE>* BackPropagate(int pTime = 0)
    {
        Tensor<DTYPE>* input = this->GetTensor();
        Tensor<DTYPE>* label = this->GetLabel()->GetResult();
        Tensor<DTYPE>* input_delta = this->GetOperator()->GetDelta();

        int batchsize = input->GetBatchSize();

        int channelsize = input->GetChannelSize();
        int rowsize = input->GetRowSize();
        int colsize = input->GetColSize();
        int capacity = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int ba = 0, i = 0; ba < batchsize; ba++)
        {
            i = ti * batchsize + ba;

            for (int j = 0, index = 0; j < capacity; j++)
            {
                index = i * capacity + j;
                (*input_delta)[index] += ((*input)[index] - (*label)[index]);
            }
        }

        return NULL;
    }

#ifdef __CUDNN__

    /*!
    @brief GPU 동작 모드에서의 MSE(Mean Squared Error) LossFunction의 순전파를 수행하는 메소드
    @param pTime 더미 변수
    @return NULL
    @ref Tensor<DTYPE>MSE::ForwardPropagate(int pTime = 0)
    */
    Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0)
    {
        this->ForwardPropagate();
        return NULL;
    }

    /*!
    @brief GPU 동작 모드에서의 MSE(Mean Squared Error) LossFunction의 역전파를 수행하는 메소드
    @param pTime 더미 변수
    @return NULL
    @ref Tensor<DTYPE>MSE::BackPropagate(int pTime = 0)
    */
    Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0)
    {
        this->BackPropagate();
        return NULL;
    }

#endif // __CUDNN__

    inline DTYPE Error(DTYPE pred, DTYPE ans) { return (pred - ans) * (pred - ans) / 2; }
};

#endif // MSE_H_
