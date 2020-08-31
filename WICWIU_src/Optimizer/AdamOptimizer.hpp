#ifndef ADAMOPTIMIZER_H_
#define ADAMOPTIMIZER_H_ value

#include "../Optimizer.hpp"

/*!
 * @class Adam Optimizer
 * @details
 */
template <typename DTYPE>
class AdamOptimizer : public Optimizer<DTYPE>
{
private:
    Container<Operator<DTYPE>*>* m_ppParameter;
    ///< 값을 업데이트 할 Tensor들을 가리키는 포인터
    Container<Tensor<DTYPE>*>* m_aaFirstVelocity;
    ///< m_Beta2의 가중치로 조정되는 gradient의 제곱과 현재 m_aaFirstVelocity 값으로 업데이트 되는
    ///< variable
    Container<Tensor<DTYPE>*>* m_aaFirstMomentum;
    ///< m_Beta1의 가중치로 조정되는 gradient와 현재 m_aaFirstVelocity 값으로 업데이트 되는 variable
    Container<Tensor<DTYPE>*>* m_aaUnbiasedVelocity;
    ///< Unbiased 작업을 거친 variable
    Container<Tensor<DTYPE>*>* m_aaUnbiasedMomentum;
    ///< Unbiased 작업을 거친 variable

    int m_numOfParameter;
    ///< 업데이트 할 Tensor의 degree

    float m_Beta1;
    ///< FirstMomentum 조정 가중치 값
    float m_Beta2;
    ///< FirstVelocity 조정 가중치 값

    float m_epsilon;
    ///< 분모 값이 0이 되는 것을 방지 하는 값

public:
    /*!
     * @brief AdamOptimizer의 생성자.
     * @details 맴버변수들을 초기화하고 Alloc 매소드를 호출한다.
     * @param *pParameterContainer
     * @param pLearningRate Optimizer의 learning rate
     * @param Beta1 FirstMomentum 조정 가중치 값
     * @param Beta2 FirstVelocity 조정 가중치 값
     * @param epsilon 분모 값이 0이 되는 것을 방지 하는 값
     * @param pOptimizeDirection Optimizing의 방향(MAXIMIZE or MINIMIZE)
     * @return 없음
     * @see int Alloc(Beta1, Beta2, epsilon)
     */
    AdamOptimizer(Container<Operator<DTYPE>*>* pParameterContainer, float pLearningRate,
                  float Beta1, float Beta2, float epsilon, OptimizeDirection pOptimizeDirection)
        : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection)
    {
#ifdef __DEBUG__
        std::cout << "AdamOptimizer::AdamOptimizer(LossFunction<DTYPE> *, float, float, float, "
                     "OptimizeDirection)"
                  << '\n';
#endif // __DEBUG__
        m_ppParameter = NULL;
        m_aaFirstVelocity = NULL;
        m_aaFirstMomentum = NULL;
        m_aaUnbiasedVelocity = NULL;
        m_aaUnbiasedMomentum = NULL;
        m_Beta1 = 0.f;
        m_Beta2 = 0.f;
        m_numOfParameter = 0;
        m_epsilon = 0.f;

        Alloc(Beta1, Beta2, epsilon);
    }

    /*!
     * @brief AdamOptimizer의 생성자.
     * @details 맴버변수들을 초기화하고 Alloc 매소드를 호출한다.
     * @param *pParameterContainer
     * @param pLearningRate Optimizer의 learning rate
     * @param Beta1 FirstMomentum 조정 가중치 값
     * @param Beta2 FirstVelocity 조정 가중치 값
     * @param epsilon 분모 값이 0이 되는 것을 방지 하는 값
     * @param weightDecayRate 가중치 매개변수가 클 때 패널티를 부과하는 값
     * @param pOptimizeDirection Optimizing의 방향(MAXIMIZE or MINIMIZE)
     * @return 없음
     * @see int Alloc(Beta1, Beta2, epsilon)
     */
    AdamOptimizer(Container<Operator<DTYPE>*>* pParameterContainer, float pLearningRate,
                  float Beta1, float Beta2, float epsilon, float weightDecayRate,
                  OptimizeDirection pOptimizeDirection)
        : Optimizer<DTYPE>(pParameterContainer, pLearningRate, weightDecayRate, pOptimizeDirection)
    {
#ifdef __DEBUG__
        std::cout << "AdamOptimizer::AdamOptimizer(LossFunction<DTYPE> *, float, float, float, "
                     "OptimizeDirection)"
                  << '\n';
#endif // __DEBUG__
        m_ppParameter = NULL;
        m_aaFirstVelocity = NULL;
        m_aaFirstMomentum = NULL;
        m_aaUnbiasedVelocity = NULL;
        m_aaUnbiasedMomentum = NULL;
        m_Beta1 = 0.f;
        m_Beta2 = 0.f;
        m_numOfParameter = 0;
        m_epsilon = 0.f;

        Alloc(Beta1, Beta2, epsilon);
    }

    /*!
     * @brief AdamOptimizer 소멸자
     * @return 없음
     */
    ~AdamOptimizer()
    {
#ifdef __DEBUG__
        std::cout << "AdamOptimizer::~AdamOptimizer()" << '\n';
#endif // __DEBUG__
        Delete();
    }

    /*!
     * @brief Optimizer의 Delete 매소드
     * @details 맴버 변수 m_aaFirstVelocity, m_aaFirstMomentum
     *       m_aaUnbiasedVelocity, m_aaUnbiasedMomentum 메모리 할당을 해제한다.
     * @return 없음
     */
    virtual void Delete()
    {
        if (m_aaFirstVelocity)
        {
            delete m_aaFirstVelocity;
            m_aaFirstVelocity = NULL;
        }

        if (m_aaFirstMomentum)
        {
            delete m_aaFirstMomentum;
            m_aaFirstMomentum = NULL;
        }

        if (m_aaUnbiasedVelocity)
        {
            delete m_aaUnbiasedVelocity;
            m_aaUnbiasedVelocity = NULL;
        }

        if (m_aaUnbiasedMomentum)
        {
            delete m_aaUnbiasedMomentum;
            m_aaUnbiasedMomentum = NULL;
        }
    }

    /*!
     * @brief Optimizer의 Alloc 매소드
     * @details 맴버 변수 m_ppParameter, m_numOfParameter,
     *       m_aaFirstVelocity, m_aaUnbiasedVelocity, m_aaFirstMomentum
     *       m_aaUnbiasedMomentum를 초기화한다.
     * @details m_aaFirstVelocity를 m_ppParameter와 같은 Shape의 Tensor를 생성하여 넣는다.
     * @details m_aaUnbiasedVelocity를 m_ppParameter와 같은 Shape의 Tensor를 생성하여 넣는다.
     * @details m_aaFirstMomentum를 m_ppParameter와 같은 Shape의 Tensor를 생성하여 넣는다.
     * @details m_aaUnbiasedMomentum를 m_ppParameter와 같은 Shape의 Tensor를 생성하여 넣는다.
     * @param Beta1 FirstMomentum 조정 가중치 값
     * @param Beta2 FirstVelocity 조정 가중치 값
     * @param epsilon Root Sqaure 값이 0이 되는 것을 방지 하는 값
     * @return 성공 시 TRUE
     * @see Container<Operator<DTYPE> *>* GetTrainableTensor()
     * @see int GetTrainableTensorDegree()
     */
    int Alloc(float Beta1, float Beta2, float epsilon)
    {
        m_aaFirstVelocity = new Container<Tensor<DTYPE>*>();
        m_aaUnbiasedVelocity = new Container<Tensor<DTYPE>*>();
        m_aaFirstMomentum = new Container<Tensor<DTYPE>*>();
        m_aaUnbiasedMomentum = new Container<Tensor<DTYPE>*>();

        Shape* pParameterGradShape = NULL;

        m_ppParameter = this->GetTrainableTensor();
        m_numOfParameter = this->GetTrainableTensorDegree();

        for (int i = 0; i < m_numOfParameter; i++)
        {
            pParameterGradShape = (*m_ppParameter)[i]->GetGradient()->GetShape();

            m_aaFirstVelocity->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            m_aaUnbiasedVelocity->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            m_aaFirstMomentum->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            m_aaUnbiasedMomentum->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));

            pParameterGradShape = NULL;
        }

        m_Beta1 = Beta1;
        m_Beta2 = Beta2;
        m_epsilon = epsilon;

        return TRUE;
    }

    /*!
     * @brief 파라미터 값들을 업데이트 하는 메소드
     * @details m_Beta1 유무에 따라 UpdateParameter 호출과 에러 메세지 호출
     * @return 성공 시 TRUE
     * @see int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pFirstMomentum,
     * Tensor<DTYPE> *pFirstVelocity, Tensor<DTYPE> *pUnbiasedMomentum, Tensor<DTYPE>
     * *pUnbiasedVelocity)
     */
    virtual int UpdateParameter()
    {
        if (m_Beta1 != 0.f)
        {
            for (int i = 0; i < m_numOfParameter; i++)
            {
                if ((*m_ppParameter)[i]->GetIsTrainable())
                    UpdateParameter((*m_ppParameter)[i], (*m_aaFirstMomentum)[i],
                                    (*m_aaFirstVelocity)[i], (*m_aaUnbiasedMomentum)[i],
                                    (*m_aaUnbiasedVelocity)[i]);
            }
        }
        else
        {
            std::cout << "Don't execute UpdateParameter On CPU" << '\n';
        }
        return TRUE;
    }

    /*!
     * @brief UpdateParameter default 함수
     * @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
     * @return 성공 시 TRUE
     */
    int UpdateParameter(Operator<DTYPE>* pParameter) { return TRUE; }

    /*!
     * @brief 파라미터 값들을 업데이트 하는 메소드
     * @details m_Beta1값으로 가중치가 조정된 pFirstMomentum와 gradinet로 pFirstMomentum를 업데이트
     * 한다.
     * @details m_Beta2값으로 가중치가 조정된 pFirstVelocity와 elementwise 연산이 된 gradient로
     * pFirstVelocity를 업데이트 한다.
     * @details 학습 초반 부, pFirstMomentum, pFirstVelocity는 0으로 biased 상태이므로 이를 unbiased
     * 해주는 연산하여 업데이트 한다.
     * @details signed_learning_rate와 pUnbiasedMomentum곱을 root가 적용된 pUnbiasedVelocity와
     * m_epsilon으로 나눈 값으로 weight(trainable_data)를 업데이트 한다.
     * @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
     * @param pFirstMomentum 업데이트 할 pFirstMomentum
     * @param pFirstVelocity 업데이트 할 pFirstVelocity
     * @param pUnbiasedMomentum 업데이트 할 pUnbiasedMomentum
     * @param pUnbiasedVelocity 업데이트 할 pUnbiasedVelocity
     * @return 성공 시 TURE
     */
    int UpdateParameter(Operator<DTYPE>* pParameter, Tensor<DTYPE>* pFirstMomentum,
                        Tensor<DTYPE>* pFirstVelocity, Tensor<DTYPE>* pUnbiasedMomentum,
                        Tensor<DTYPE>* pUnbiasedVelocity)
    {
        Tensor<DTYPE>* trainable_data = pParameter->GetResult();
        Tensor<DTYPE>* gradient = pParameter->GetGradient();

        float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++)
        {
            (*pFirstMomentum)[i] =
                (m_Beta1 * (*pFirstMomentum)[i]) + ((1.f - m_Beta1) * (*gradient)[i]);
            (*pFirstVelocity)[i] = (m_Beta2 * (*pFirstVelocity)[i]) +
                                   ((1.f - m_Beta2) * ((*gradient)[i] * (*gradient)[i]));
            (*pUnbiasedMomentum)[i] = (*pFirstMomentum)[i] / (1.f - m_Beta1);
            (*pUnbiasedVelocity)[i] = (*pFirstVelocity)[i] / (1.f - m_Beta2);
            (*trainable_data)[i] += ((signed_learning_rate * (*pUnbiasedMomentum)[i]) /
                                     (std::sqrt((*pUnbiasedVelocity)[i]) + m_epsilon));
        }

        return TRUE;
    }

#ifdef __CUDNN__

    /*!
     * @brief m_aaFirstMomentum, m_aaFirstVelocity,
     *     m_aaUnbiasedMomentum, m_aaUnbiasedVelocity Tensor의 device를 idOfDevice번째 GPU로 바꾼다.
     * @param idOfDevice 사용 하는 GPU번호
     */
    void InitializeAttributeForGPU(unsigned int idOfDevice)
    {
        if (m_Beta1 != 0.f)
        {
            for (int i = 0; i < m_numOfParameter; i++)
            {
                (*m_aaFirstMomentum)[i]->SetDeviceGPU(idOfDevice);
                (*m_aaFirstVelocity)[i]->SetDeviceGPU(idOfDevice);
                (*m_aaUnbiasedMomentum)[i]->SetDeviceGPU(idOfDevice);
                (*m_aaUnbiasedVelocity)[i]->SetDeviceGPU(idOfDevice);
            }
        }
    }

    /*!
     * @brief 파라미터 값들을 업데이트 하는 메소드(GPU)
     * @details m_Beta1 유무에 따라 UpdateParameterOnGPU 호출과 에러 메세지 호출
     * @return 성공 시 TRUE
     * @see int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pFirstMomentum,
     * Tensor<DTYPE> *pFirstVelocity, Tensor<DTYPE> *pUnbiasedMomentum, Tensor<DTYPE>
     * *pUnbiasedVelocity)
     */
    virtual int UpdateParameterOnGPU()
    {
        if (m_Beta1 != 0.f)
        {
            for (int i = 0; i < m_numOfParameter; i++)
            {
                if ((*m_ppParameter)[i]->GetIsTrainable())
                    UpdateParameterOnGPU((*m_ppParameter)[i], (*m_aaFirstMomentum)[i],
                                         (*m_aaFirstVelocity)[i], (*m_aaUnbiasedMomentum)[i],
                                         (*m_aaUnbiasedVelocity)[i]);
            }
        }
        else
        {
            std::cout << "Don't execute UpdataParameterOnGPU" << '\n';
        }

        return TRUE;
    }

    /*!
     * @brief UpdateParameterOnGPU default 함수
     * @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
     * @return 성공 시 TRUE
     */
    int UpdateParameterOnGPU(Operator<DTYPE>* pParameter) { return TRUE; }

    /*!
     * @brief 파라미터 값들을 업데이트 하는 GPU 메서드.
     * @details 실행시 해당 cu 파일의 __global__함수 실행.
     * @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
     * @param pFirstMomentum 업데이트 할 pFirstMomentum
     * @param pFirstVelocity 업데이트 할 pFirstVelocity
     * @param pUnbiasedMomentum 업데이트 할 pUnbiasedMomentum
     * @param pUnbiasedVelocity 업데이트 할 pUnbiasedVelocity
     * @see __global__ void AdagradUpdate_kernel(float *pDevWeight, float *pDevAccGradient, int
     * weightDim, float signed_learning_rate, float epsilon, float weightDecayRate, float
     * *pDevGradientSquared)
     * @return 성공 시 TURE
     */
    int UpdateParameterOnGPU(Operator<DTYPE>* pParameter, Tensor<DTYPE>* pFirstMomentum,
                             Tensor<DTYPE>* pFirstVelocity, Tensor<DTYPE>* pUnbiasedMomentum,
                             Tensor<DTYPE>* pUnbiasedVelocity);

#endif // if __CUDNN__
};

#endif // ADAMOPTIMIZER_H_
