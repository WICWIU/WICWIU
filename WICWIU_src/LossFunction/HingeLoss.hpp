#ifndef HINGELOSS_H_
#define HINGELOSS_H_ value

#include "../LossFunction.hpp"

/*!
@class HingeLoss Hinge Loss Metric를 이용해 뉴럴 네트워크의 손실 함수를 계산하는 클래스
@details Hinge Loss 계산 식을 이용해 뉴럴 네트워크의 순전파를 통해 계산된 출력 Tensor와 레이블 값의
손실 함수를 계산한다
*/
template <typename DTYPE>
class HingeLoss : public LossFunction<DTYPE>
{
private:
    Tensor<DTYPE>* m_aindexForBackProp;
    ///< 역전파 메소드의 인덱스로 사용하기 위한 더미 Tensor
    float m_theta;
    ///< Hinge Loss 수식의 Threshold에 해당하는 값

public:
    /*!
    @brief HingeLoss LossFunction 클래스 생성자
    @details LossFunction 클래스의 생성자를 호출하고, Operator와 theta을 매개변수로 전달하여
    HingeLoss<DTYPE>::Alloc(Operator<DTYPE> *pOperator, float theta) 메소드를 호출한다.
    @param pOperator HingeLoss<DTYPE>::Alloc(Operator<DTYPE> *pOperator, float theta) 메소드의
    매개변수로 전달할 Operator
    @param pLabel LossFunction의 입력 레이블에 해당하는 Operator
    @param theta alloc 메소드의 theta 값으로 전달할 파라미터, 값을 지정하지 않을 시 1.f로 초기화
    @return 없음
    @see HingeLoss<DTYPE>::Alloc(Operator<DTYPE> *pOperator, float theta)
    */
    HingeLoss(Operator<DTYPE>* pOperator, Operator<DTYPE>* pLabel, float theta = 1.f)
        : LossFunction<DTYPE>(pOperator, pLabel)
    {
#ifdef __DEBUG__
        std::cout << "HingeLoss::HingeLoss(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
#endif // __DEBUG__
        this->Alloc(pOperator, theta);
    }

    /*!
    @brief HingeLoss LossFunction 클래스 생성자
    @details LossFunction 클래스의 생성자를 호출하고, Operator와 1.f에 해당하는 theta 값 매개변수로
    전달하여 HingeLoss<DTYPE>::Alloc(Operator<DTYPE> *pOperator, float theta) 메소드를 호출한다.
    @param pOperator HingeLoss<DTYPE>::Alloc(Operator<DTYPE> *pOperator, float theta) 메소드의
    매개변수로 전달할 Operator
    @param pLabel LossFunction의 입력 레이블에 해당하는 Operator
    @param pName LossFunction의 이름
    @return 없음
    @see HingeLoss<DTYPE>::Alloc(Operator<DTYPE> *pOperator, float theta)
    */
    HingeLoss(Operator<DTYPE>* pOperator, Operator<DTYPE>* pLabel, std::string pName)
        : LossFunction<DTYPE>(pOperator, pLabel, pName)
    {
#ifdef __DEBUG__
        std::cout << "HingeLoss::HingeLoss(Operator<DTYPE> *, Operator<DTYPE> *, std::string)"
                  << '\n';
#endif // __DEBUG__
        this->Alloc(pOperator, 1.f);
    }

    /*!
    @brief HingeLoss LossFunction 클래스 생성자
    @details LossFunction 클래스의 생성자를 호출하고, Operator와 theta을 매개변수로 전달하여
    HingeLoss<DTYPE>::Alloc(Operator<DTYPE> *pOperator, float theta) 메소드를 호출한다.
    @param pOperator HingeLoss<DTYPE>::Alloc(Operator<DTYPE> *pOperator, float theta) 메소드의
    매개변수로 전달할 Operator
    @param pLabel LossFunction의 입력 레이블에 해당하는 Operator
    @param theta alloc 메소드의 theta 값으로 전달할 파라미터
    @see HingeLoss<DTYPE>::Alloc(Operator<DTYPE> *pOperator, float theta)
    */
    HingeLoss(Operator<DTYPE>* pOperator, Operator<DTYPE>* pLabel, float theta, std::string pName)
        : LossFunction<DTYPE>(pOperator, pLabel, pName)
    {
#ifdef __DEBUG__
        std::cout << "HingeLoss::HingeLoss(Operator<DTYPE> *, Operator<DTYPE> *, std::string)"
                  << '\n';
#endif // __DEBUG__
        this->Alloc(pOperator, theta);
    }

    /*!
    @brief HingeLoss LossFunction 클래스 소멸자
    @return 없음
    */
    ~HingeLoss()
    {
#ifdef __DEBUG__
        std::cout << "HingeLoss::~HingeLoss()" << '\n';
#endif // __DEBUG__
    }

    /*!
    @brief HingeLoss Lossfunction의 멤버 변수들을 동적 할당하는 메소드
    @details 매개변수로 전달받은 Operator를 Input Operator에 할당하고 초기화 된 Result 텐서를
    동적으로 할당 및 생성한다.
    @details 역전파를 위한 인덱스 더미 텐서를 동적으로 할당 및 생성하고 theta 값을 초기화한다.
    @param pOperator CrossEntropy LossFunction의 입력에 해당하는 Operator
    @param theta LossFunction의 멤버 변수 theta에 할당할 값
    @return TRUE
    */
    int Alloc(Operator<DTYPE>* pOperator, float theta)
    {
#ifdef __DEBUG__
        std::cout << "HingeLoss::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
#endif // __DEBUG__

        Operator<DTYPE>* pInput = pOperator;

        int timesize = pInput->GetResult()->GetTimeSize();
        int batchSize = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize = pInput->GetResult()->GetRowSize();
        int colsize = pInput->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchSize, 1, 1, 1));

        m_aindexForBackProp =
            Tensor<DTYPE>::Constants(timesize, batchSize, channelsize, rowsize, colsize, 0.f);

        m_theta = theta;

        return TRUE;
    }

    /*!
    @brief LossFunction 클래스의 메모리를 할당 해제하는 메소드
    @details Index for BackPropagation Tensor가 존재할 경우 Tensor의 메모리를 할당 해제하고 0으로
    초기화한다.
    @return 없음
    */
    void Delete()
    {
        if (m_aindexForBackProp)
        {
            delete m_aindexForBackProp;
            m_aindexForBackProp = NULL;
        }
    }

    /*!
    @brief GPU 동작 모드에서의 HingeLoss LossFunction의 순전파를 수행하는 메소드
    @details 구성한 뉴럴 네트워크에서 얻어진 결과 값을 레이블 값과 비교해 Hinge Loss 값을 구한다
    @param timeIdx Time 축의 인덱스, 미지정 시 0으로 초기화
    @return 뉴럴 네트워크 결과 값에 대한 Hinge Loss
    */
    Tensor<DTYPE>* ForwardPropagate(int timeIdx = 0)
    {
        Tensor<DTYPE>* input = this->GetTensor();
        Tensor<DTYPE>* desired = this->GetLabel()->GetResult();
        Tensor<DTYPE>* result = this->GetResult();
        m_aindexForBackProp->Reset();

        int batchSize = input->GetBatchSize();

        int channelsize = input->GetChannelSize();
        int rowsize = input->GetRowSize();
        int colsize = input->GetColSize();
        int outputDim = channelsize * rowsize * colsize;

        int trueClass = 0;
        DTYPE trueClassScore = (DTYPE)0;

        for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
        {
            int globalSampleIdx = (timeIdx * batchSize + sampleIdx);

            DTYPE* dStart = &(*desired)[globalSampleIdx * outputDim];
            DTYPE* dLimit = dStart + outputDim;

            for (DTYPE* dp = dStart; dp < dLimit; dp++)
            {
                if (*dp == (DTYPE)1)
                {
                    trueClass = dp - dStart;
                    break;
                }
            }

            int firstOutputIdx = globalSampleIdx * outputDim;
            trueClassScore = (*input)[firstOutputIdx + trueClass];

            DTYPE* rp = &(*result)[globalSampleIdx]; // output
            DTYPE* ip = &(*input)[firstOutputIdx];   // input index
            int curClass = 0;

            for (DTYPE* dp = dStart; dp < dLimit; dp++, curClass++, ip++)
            {
                if ((*ip + m_theta > trueClassScore) && (curClass != trueClass))
                {
                    *rp += *ip - trueClassScore + m_theta;

                    // for backpropagation, not necessary for forward
                    (*m_aindexForBackProp)[curClass + firstOutputIdx] += 1;
                    (*m_aindexForBackProp)[trueClass + firstOutputIdx] += -1;
                }
            }
        }

        return result;
    }

    /*!
    @brief GPU 동작 모드에서의 HingeLoss LossFunction의 역전파를 수행하는 메소드
    @details 구성한 뉴럴 네트워크에서 얻어진 HingeLoss LossFunction에 대한 입력 Tensor의 Gradient를
    계산한다
    @param pTIme Time 축의 인덱스, 미지정 시 0으로 초기화
    @return NULL
    */
    Tensor<DTYPE>* BackPropagate(int pTime = 0)
    {
        Tensor<DTYPE>* input = this->GetTensor();
        Tensor<DTYPE>* label = this->GetLabel()->GetResult();
        Tensor<DTYPE>* input_delta = this->GetOperator()->GetDelta();

        int batchSize = input->GetBatchSize();

        int channelsize = input->GetChannelSize();
        int rowsize = input->GetRowSize();
        int colsize = input->GetColSize();
        int capacity = channelsize * rowsize * colsize;

        int ti = pTime;

        int temp = 0;

        for (int ba = 0, i = 0; ba < batchSize; ba++)
        {
            i = ti * batchSize + ba;

            for (int j = 0, index = 0; j < capacity; j++)
            {
                index = i * capacity + j;

                (*input_delta)[index] += (*m_aindexForBackProp)[index];
            }
        }

        return NULL;
    }

#ifdef __CUDNN__

    /*!
    @brief GPU 동작 모드에서의 HingeLoss LossFunction의 순전파를 수행하는 메소드
    @param pTime 더미 파라미터
    @return NULL
    @see Tensor<DTYPE>HingeLoss::ForwardPropagate(int timeIdx = 0)
    */
    Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0)
    {
        this->ForwardPropagate();
        return NULL;
    }

    /*!
    @brief GPU 동작 모드에서의 HingeLoss LossFunction의 역전파를 수행하는 메소드
    @param pTime 더미 파라미터
    @return NULL
    @see Tensor<DTYPE>HingeLoss::BackPropagate(int pTime = 0)
    */
    Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0)
    {
        this->BackPropagate();
        return NULL;
    }

#endif // __CUDNN__
};

#endif // HINGELOSS_H_
