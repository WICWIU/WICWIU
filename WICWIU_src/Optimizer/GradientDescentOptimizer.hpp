#ifndef GRADIENTDESCENTOPTIMIZER_H_
#define GRADIENTDESCENTOPTIMIZER_H_    value

#include "../Optimizer.hpp"

template<typename DTYPE> class GradientDescentOptimizer : public Optimizer<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_ppParameter;
    ///< 값을 업데이트 할 Tensor들을 가리키는 포인터
    Container<Tensor<DTYPE> *> *m_aaVelocity;
    ///<  momentum의 누적된 속도

    int m_numOfParameter;
    ///< 업데이트 할 Tensor의 degree
    float m_momentum;
    ///< Optimizer의 momentum 값

public:
    /*!
    @brief GradientDescentOptimizer의 생성자.
    @details 맴버변수들을 초기화하고 Alloc 매소드를 호출한다.
    @param *pParameterContainer
    @param pLearningRate Optimizer의 learning rate
    @param pOptimizeDirection Optimizing의 방향(MAXIMIZE or MINIMIZE)
    @return 없음
    @see int Alloc()
    */
    GradientDescentOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc();
    }

    /*!
    @brief GradientDescentOptimizer의 생성자.
    @details 맴버변수들을 초기화하고 momentum값을 파라미터로 하는 Alloc 매소드를 호출한다.
    @param pParameterContainer
    @param pLearningRate Optimizer의 learning rate
    @param momentum Optimize의 momentum
    @param pOptimizeDirection Optimizing의 방향(MAXIMIZE or MINIMIZE)
    @return 없음
    @see int Alloc(float momentum)
    */
    GradientDescentOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float momentum, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc(momentum);
    }

    /*!
    @brief GradientDescentOptimizer의 생성자.
    @details 맴버변수들을 초기화하고 momentum값을 파라미터로 하는 Alloc 매소드를 호출한다.
    @param pParameterContainer
    @param pLearningRate Optimizer의 learning rate
    @param momentum Optimize의 momentum
    @param weightDecayRate
    @paramp OptimizeDirection Optimizing의 방향(MAXIMIZE or MINIMIZE)
    @return 없음
    @see int Alloc(float momentum)
    */
    GradientDescentOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float momentum, float weightDecayRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, weightDecayRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "GradientDescentOptimizer::GradientDescentOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc(momentum);
    }

    /*!
    @brief GradientDescentOptimizer의 소멸자
    @return 없음
    */
    ~GradientDescentOptimizer() {
        #ifdef __DEBUG__
        std::cout << "GradientDescentOptimizer::~GradientDescentOptimizer()" << '\n';
        #endif  // __DEBUG__
    }

    /*!
    @brief Optimizer의 Alloc 매소드
    @details 맴버 변수 m_ppParameter와 m_numOfParameter를 초기화한다.
    @return 성공 시 TRUE
    @see Container<Operator<DTYPE> *>* GetTrainableTensor()
    @see int GetTrainableTensorDegree()
    */
    int Alloc() {
        m_ppParameter    = this->GetTrainableTensor();
        m_numOfParameter = this->GetTrainableTensorDegree();

        return TRUE;
    }

    /*!
    @brief Optimizer의 Alloc 매소드
    @details 맴버 변수 m_aaVelocity, m_momentum를 초기화 한다.
    @details m_aaVelocity에 m_ppParameter와 같은 Shape의 Tensor를 생성하여 넣는다.
    @param momentum Optimizer의 monentum값
    @return 성공 시 TRUE
    @see int Alloc()
    */
    int Alloc(float momentum) {
        Alloc();
        m_aaVelocity = new Container<Tensor<DTYPE> *>();

        Shape *pParameterGradShape = NULL;

        for (int i = 0; i < m_numOfParameter; i++) {
            pParameterGradShape = (*m_ppParameter)[i]->GetGradient()->GetShape();

            // std::cout << (*m_ppParameter)[i]->GetName() << '\n';
            // std::cout << pParameterGradShape << '\n';

            m_aaVelocity->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));
            pParameterGradShape = NULL;
        }

        m_momentum = momentum;

        return TRUE;
    }

    /*!
    @brief m_aaVelocity내부의 Tensor의 device를 idOfDevice번째 GPU로 바꾼다.
    @param idOfDevice 사용 하는 GPU번호
    */
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        if (m_momentum != 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                (*m_aaVelocity)[i]->SetDeviceGPU(idOfDevice);
            }
        }
    }

    /*!
    @brief 파라미터 값들을 업데이트 하는 메소드
    @details m_momentum값에 따라 다른 UpdataParameter를 호출한다.
    @return 성공 시 TRUE
    @see int UpdateParameter(Operator<DTYPE> *pParameter)
    @see int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity)
    */
    virtual int UpdateParameter() {
        if (m_momentum == 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameter((*m_ppParameter)[i]);
            }
        } else {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameter((*m_ppParameter)[i], (*m_aaVelocity)[i]);
            }
        }

        return TRUE;
    }

    /*!
    @brief 파라미터 값들을 업데이트 하는 메소드
    @details Parameter안에 있는 Tensor의 새로 계산된 gradinet값과 learning_rate의 곱, weightDecayRate와 기존 weight(trainable_date)의 곱으로 weight(trainable_date)값을 업데이트한다.
    @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @return 성공 시 TRUE
    */
    int UpdateParameter(Operator<DTYPE> *pParameter) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float learning_rate   = this->GetOptimizeDirection() * this->GetLearningRate();
        float weightDecayRate = this->GetWeightDecayRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*trainable_data)[i] += (learning_rate * (*gradient)[i] + weightDecayRate * (*trainable_data)[i]);
        }

        return TRUE;
    }

    /*!
    @brief 파라미터 값들을 업데이트 하는 메소드
    @details Parameter안에 있는 Tensor의 새로 계산된 gradinet값과 learning_rate의 곱, weightDecayRate와 기존 weight(trainable_date)의 곱으로 weight(trainable_date)값을 업데이트한다.
    @details momentum과 pVelocity의 곱과 learnung_rate와 gradient의 곱으로 pVelocity의 값을 업데이트 한다.
    @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @param pVelocity 업데이트 할 pVelocity
    @return 성공 시 TURE
    */
    int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        float learning_rate   = this->GetOptimizeDirection() * this->GetLearningRate();
        float weightDecayRate = this->GetWeightDecayRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*pVelocity)[i]      = m_momentum * (*pVelocity)[i] + learning_rate * (*gradient)[i];
            (*trainable_data)[i] += ((*pVelocity)[i] + weightDecayRate * (*trainable_data)[i]);
        }

        return TRUE;
    }

#ifdef __CUDNN__


    /*!
    @brief GPU의 메모리에 있는 Parameter값들을 업데이트 하는 메소드.
    @details momentum여부에 따라 오버로딩 된 각기 다른 UpdateParameterOnGPU 메소드를 호출함.
    @return 성공 시 TRUE.
    */
    virtual int UpdateParameterOnGPU() {
        if (m_momentum == 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameterOnGPU((*m_ppParameter)[i]);
            }
        } else {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameterOnGPU((*m_ppParameter)[i], (*m_aaVelocity)[i]);
            }
        }

        return TRUE;
    }

    /*!
    @brief momentum없이 GPU의 메모리에 있는 Parameter값들을 업데이트 하는 메소드.
    @details 파라미터로 전달받은 Operator의 Result와 Gradient값을 업데이트한다.
    @details cudnnAddTensor를 이용하여 Gradient값을 더한다.
    @param pParameter Gradient값을 업데이트 할 Operator.
    @return 성공 시 TRUE
    */
    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        cudnnTensorDescriptor_t dataDesc = trainable_data->GetDescriptor();
        cudnnTensorDescriptor_t gradDesc = gradient->GetDescriptor();

        DTYPE *m_pDevData = trainable_data->GetGPUData();
        DTYPE *m_pDevGrad = gradient->GetGPUData();

        float learning_rate   = this->GetOptimizeDirection() * this->GetLearningRate();
        float weightDecayRate = this->GetWeightDecayRate();

        float alpha = 1.f + learning_rate * weightDecayRate;
        float beta  = learning_rate;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &beta, gradDesc, m_pDevGrad,
                                  &alpha, dataDesc, m_pDevData));

        return TRUE;
    }

    /*!
    @brief momentum을 적용하여 GPU의 메모리에 있는 Parameter값들을 업데이트 하는 메소드.
    @details 파라미터로 전달 받은 Operator의 Result와 Gradient값을 momentum을 반영하여 업데이트한다.
    @param pParameter Gradient값을 업데이트 할 Operator.
    @param pVelocity momentum값.
    @return 성공 시 TRUE.
    */
    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();

        cudnnTensorDescriptor_t dataDesc = trainable_data->GetDescriptor();
        cudnnTensorDescriptor_t gradDesc = gradient->GetDescriptor();
        cudnnTensorDescriptor_t veloDesc = pVelocity->GetDescriptor();

        DTYPE *m_pDevData = trainable_data->GetGPUData();
        DTYPE *m_pDevGrad = gradient->GetGPUData();
        DTYPE *m_pDevVelo = pVelocity->GetGPUData();

        float learning_rate   = this->GetOptimizeDirection() * this->GetLearningRate();
        float weightDecayRate = this->GetWeightDecayRate();

        float alpha = 1.f + learning_rate * weightDecayRate;
        // std::cout << "weight decay  : "<< weightDecayRate << '\n';
        // std::cout << "alpha : " << alpha << '\n';
        float beta  = learning_rate;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &beta, gradDesc, m_pDevGrad,
                                  &m_momentum, veloDesc, m_pDevVelo));

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &alpha, veloDesc, m_pDevVelo,
                                  &alpha, dataDesc, m_pDevData));

        return TRUE;
    }

#endif  // if __CUDNN__
};


#endif  // GRADIENTDESCENTOPTIMIZER_H_
