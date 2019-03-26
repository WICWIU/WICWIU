#ifndef RMSPROPOPTIMIZER_H_
#define RMSPROPOPTIMIZER_H_   value

/*
============================논문 버젼 ========================================================
reference :
https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
rmsprop algorithm [tieleman2012rmsprop]

explain:
centered version보다는 Accuracy측면에서는 성능이 떨어지나 속도가 빠름.
A detailed description of rmsprop.
- maintain a moving (discounted) average of the square of gradients
- divide gradient by the root of this average

m_aaMeanSqueared = decay * mean_square{t-1} + (1 - decay) * gradient ** 2

trainable_data = momentum * trainable_data{t-1} + learning_rate * g_t / sqrt(m_aaMeanSqueared + epsilon)

delta = - trainable_data

==========================centered version=======================================================
reference :
https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
https://arxiv.org/pdf/1308.0850v5.pdf //논문

notice :
centered version 사용시 반드시 learning rate는 0.0001로 주어야 한다. 논문참조
값이 크면 학습이 잘 안된다.

explain:
 Accuracy 측면에서는 좋으나, 수식이 많아 속도감이 떨어짐.

If True, gradients are normalized by the estimated variance of the gradient;
if False, by the uncentered second moment. Setting this to True may help with Train,
but is slightly more expensive in terms of computation and memory. Defaults to False.
This implementation of RMSProp uses plain momentum, not Nesterov momentum.
The centered version additionally maintains a moving (discounted) average of the
gradients, and uses that average to estimate the variance:

mean_grad = decay * mean_square{t-1} + (1-decay) * gradient

m_aaMeanSqueared  = decay * mean_square{t-1} + (1-decay) * gradient ** 2

trainable_data = momentum * mom{t-1} + learning_rate * g_t /  sqrt(mean_square - mean_grad**2 + epsilon)

delta = - mom

==================================================================================
*/

#include "../Optimizer.hpp"


template<typename DTYPE> class RMSPropOptimizer : public Optimizer<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_ppParameter;
    ///< 값을 업데이트 할 Tensor들을 가리키는 포인터
    Container<Tensor<DTYPE> *> *m_aaMeanSquared;  //  = decay * mean_square{t-1} + (1-decay) * gradient ** 2
    ///< decay 가중치로 조정된  gradient의 제곱과 현재 m_aaMeanSquared 값으로 업데이트 되는 variable
    Container<Tensor<DTYPE> *> *m_aaMeanGrad;
    ///< decay 가중치로 조정된  gradient와 현재 m_aaMeanSquared 값으로 업데이트 되는 variable

    int m_numOfParameter;
    ///< 업데이트 할 Tensor의 degree

    float m_decay; //decay = 0.9
    ///< m_aaMeanSquared, m_aaMeanGrad, gradient 조정 가중치 값
    float m_epsilon; //epsilon = 0.001
    ///< 분모 값이 0이 되는 것을 방지 하는 값

    bool m_centered;
    ///< cemtered version enable boolean 변수

public:
    /*!
    @brief RMSPropOptmizer 생성자.
    @details 맴버변수들을 초기화하고 Alloc 매소드를 호출한다.
    @param *pParameterContainer 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @param pLearningRate Optimizer의 learning rate
    @param decay m_aaMeanSquared, m_pMeanGrad, m_aaMeanGrad, gradient 조정 가중치 값
    @param epsilon 분모 값이 0이 되는 것을 방지 하는 값
    @param centered cemtered version enable boolean 변수
    @param pOptimizeDirection Optimizing의 방향(MAXIMIZE or MINIMIZE)
    @return 없음
    @see int Alloc(decay, epsilon, centered)
    */
    RMSPropOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float decay, float epsilon, bool centered, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "RMSPropOptimizer::RMSPropOptimizer(LossFunction<DTYPE> *, float, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter       = NULL;
        m_aaMeanSquared     = NULL;
        m_aaMeanGrad        = NULL;
        m_numOfParameter    = 0;
        m_decay             = 0.f;
        m_epsilon           = 0.f;
        m_centered          = FALSE;

        Alloc(decay, epsilon, centered);
    }

    /*!
    @brief RMSPropOptmizer 생성자.
    @details 맴버변수들을 초기화하고 Alloc 매소드를 호출한다.
    @param *pParameterContainer 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @param pLearningRate Optimizer의 learning rate
    @param decay m_aaMeanSquared, m_pMeanGrad, m_aaMeanGrad, gradient 조정 가중치 값
    @param epsilon 분모 값이 0이 되는 것을 방지 하는 값
    @param weightDecayRate 가중치 매개변수가 클 때 패널티를 부과하는 값
    @param pOptimizeDirection Optimizing의 방향(MAXIMIZE or MINIMIZE)
    @return 없음
    @see int Alloc(decay, epsilon, centered)
    */
    RMSPropOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float decay, float epsilon, bool centered, float weightDecayRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, weightDecayRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "RMSPropOptimizer::RMSPropOptimizer(LossFunction<DTYPE> *, float, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter       = NULL;
        m_aaMeanSquared     = NULL;
        m_aaMeanGrad        = NULL;
        m_numOfParameter    = 0;
        m_decay             = 0.f;
        m_epsilon           = 0.f;
        m_centered          = FALSE;

        Alloc(decay, epsilon, centered);
    }

    /*!
    @brief RMSPropOpmitzer 소멸자
    @return 없음
    */
    ~RMSPropOptimizer() {
        #ifdef __DEBUG__
        std::cout << "RMSPropOptimizer::~RMSPropOptimizer()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    /*!
    @brief Optimizer의 Delete 매소드
    @details 맴버 변수 m_aaMeanSquared, m_aaMeanGrad의 메모리 할당을 해제한다.
    @return 없음
    */
    virtual void Delete() {
        if (m_aaMeanSquared) {
            delete m_aaMeanSquared;
            m_aaMeanSquared = NULL;
        }
        if (m_aaMeanGrad) {
            delete m_aaMeanGrad;
            m_aaMeanGrad = NULL;
        }
    }



    /*!
    @brief Optimizer의 Alloc 매소드
    @details 맴버 변수 m_ppParameter, m_numOfParameter, m_aaMeanSquared, m_aaMeanGrad를 초기화한다.
    @details m_aaMeanSquared를 m_ppParameter와 같은 Shape의 Tensor를 생성하여 넣는다.
    @details m_aaMeanGrad를 m_ppParameter와 같은 Shape의 Tensor를 생성하여 넣는다.
    @param decay m_aaMeanSquared, m_pMeanGrad, m_aaMeanGrad, gradient 조정 가중치 값
    @param epsilon Root Sqaure 값이 0이 되는 것을 방지 하는 값
    @param centered cemtered version enable boolean 변수
    @return 성공 시 TRUE
    @see Container<Operator<DTYPE> *>* GetTrainableTensor()
    @see int GetTrainableTensorDegree()
    */
    int Alloc(float decay, float epsilon, bool centered) {

        m_ppParameter    = this->GetTrainableTensor();
        m_numOfParameter = this->GetTrainableTensorDegree();

        m_aaMeanSquared = new Container<Tensor<DTYPE> *>();
        m_aaMeanGrad    = new Container<Tensor<DTYPE> *>();

        Shape *pParameterGradShape = NULL;

        for (int i = 0; i < m_numOfParameter; i++) {
            pParameterGradShape = (*m_ppParameter)[i]->GetGradient()->GetShape();
            m_aaMeanSquared->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            m_aaMeanGrad->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            pParameterGradShape = NULL;
        }

        m_decay = decay;
        m_epsilon = epsilon;
        m_centered = centered;

        return TRUE;
    }

    /*!
    @brief 파라미터 값들을 업데이트 하는 메소드
    @details m_centered 유무에 따라 centered version 과 not use RMSProp UpdateParameter 호출
    @return 성공 시 TRUE
    @see int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *m_pMeanSquared)
    @see int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *m_pMeanSquared, Tensor<DTYPE> *m_pMeanGrad)
    */
    virtual int UpdateParameter() {

        if(m_centered == TRUE){
            for (int i = 0; i < m_numOfParameter; i++) {
                if((*m_ppParameter)[i]->GetIsTrainable()) UpdateParameter((*m_ppParameter)[i], (*m_aaMeanSquared)[i], (*m_aaMeanGrad)[i]);
            }
        }else{
            for (int i = 0; i < m_numOfParameter; i++) {
                if((*m_ppParameter)[i]->GetIsTrainable()) UpdateParameter((*m_ppParameter)[i], (*m_aaMeanSquared)[i]);
            }
        }
        return TRUE;
    }

    /*!
    @brief UpdateParameter default 함수
    @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @return 성공 시 TRUE
    */
    int UpdateParameter(Operator<DTYPE> *pParameter){
      return TRUE;
    }

    /*!
    @brief 파라미터 값들을 업데이트 하는 메소드
    @details m_decay 가중치로 조정된 m_pMeanSquared, gradinet로 m_pMeanSquared 업데이트 한다.
    @details 업데이트 된 m_pMeanSquared로 지수평균 이동 공식을 적용하여 파라미터를 업데이트 한다.
    @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @param pMeanSquared 업데이트 할 pMeanSquared
    @return 성공 시 TURE
    */
    int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *m_pMeanSquared) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();//minimizer = -1
        int capacity = trainable_data->GetCapacity();


        for (int i = 0; i < capacity; i++) {
            (*m_pMeanSquared)[i]       = (m_decay * (*m_pMeanSquared)[i]) + (( 1.f - m_decay) * ((*gradient)[i] * (*gradient)[i]));
            (*trainable_data)[i]    += ((signed_learning_rate * (*gradient)[i]) / std::sqrt((*m_pMeanSquared)[i] + m_epsilon));
        }

        return TRUE;
    }

    /*!
    @brief 파라미터 값들을 업데이트 하는 메소드
    @details m_decay 가중치로 조정된 gradient로 m_pMeanGrad를 업데이트한다.
    @details m_decay 가중치로 조정된 m_pMeanSquared, gradinet로 m_pMeanSquared 업데이트 한다.
    @details 업데이트 된 m_pMeanSquared와 업데이트 된 m_pMeanGrad의 제곱의 차를 지수평균이동공식에 적용하여 파라미터를 업데이트 한다.
    @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @param pMeanSquared 업데이트 할 pMeanSquared
    @param pMeanGrad 업데이트 할 pMeanGrad
    @return 성공 시 TURE
    */
    int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *m_pMeanSquared, Tensor<DTYPE> *m_pMeanGrad){
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float signed_learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();//minimizer = -1
        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*m_pMeanGrad)[i]     = (m_decay * (*m_pMeanGrad)[i]) + ((1.f - m_decay) * (*gradient)[i]);
            (*m_pMeanSquared)[i]  = (m_decay * (*m_pMeanSquared)[i]) + (( 1.f - m_decay) * ((*gradient)[i] * (*gradient)[i]));
            (*trainable_data)[i] += ((signed_learning_rate * (*gradient)[i]) / std::sqrt(((*m_pMeanSquared)[i] - ((*m_pMeanGrad)[i] * (*m_pMeanGrad)[i]))+ m_epsilon));
        }

        return TRUE;
    }

#ifdef __CUDNN__

    /*!
    @brief m_aaMeanSquared, m_aaMeanGrad Tensor의 device를 idOfDevice번째 GPU로 바꾼다.
    @param idOfDevice 사용 하는 GPU번호
    */
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
            for (int i = 0; i < m_numOfParameter; i++) {
                (*m_aaMeanSquared)[i]->SetDeviceGPU(idOfDevice);
                (*m_aaMeanGrad)[i]->SetDeviceGPU(idOfDevice);
            }
    }

    /*!
    @brief 파라미터 값들을 업데이트 하는 메소드(GPU)
    @details m_centered 유무에 따라 centered version 과 not use RMSProp UpdateParameterOnGPU 호출
    @return 성공 시 TRUE
    @see int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *m_pMeanSquared)
    @see int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *m_pMeanSquared, Tensor<DTYPE> *m_pMeanGrad)
    */
    virtual int UpdateParameterOnGPU() {
        if (m_centered == TRUE) {
            for (int i = 0; i < m_numOfParameter; i++) {
              if((*m_ppParameter)[i]->GetIsTrainable()) UpdateParameterOnGPU((*m_ppParameter)[i], (*m_aaMeanSquared)[i], (*m_aaMeanGrad)[i]);
            }
        }else{
            for (int i = 0; i < m_numOfParameter; i++) {
              if((*m_ppParameter)[i]->GetIsTrainable()) UpdateParameterOnGPU((*m_ppParameter)[i], (*m_aaMeanSquared)[i]);
          }
        }

        return TRUE;
    }


    /*!
    @brief UpdateParameterOnGPU default 함수
    @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @return 성공 시 TRUE
    */
    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter){
      return TRUE;
    }

    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *m_pMeanSquared);

    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *m_pMeanSquared, Tensor<DTYPE> *m_pMeanGrad);



#endif  // if __CUDNN__
 };


#endif  // RMSPROPOPTIMIZER_H_
