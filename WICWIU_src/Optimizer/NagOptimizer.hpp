#ifndef NAGOPTIMIZER_H_
#define NAGOPTIMIZER_H_    value

#include "../Optimizer.hpp"

template<typename DTYPE> class NagOptimizer : public Optimizer<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_ppParameter;
    ///< 값을 업데이트 할 Tensor들을 가리키는 포인터
    Container<Tensor<DTYPE> *> *m_aaVelocity;
    ///< 이동 벡터 variable

    int m_numOfParameter;
    ///< 업데이트 할 Tensor의 degree
    float m_momentum;
    ///< step size 조정 값

public:

    /*!
    @brief NagOptimizer의 생성자.
    @details 맴버변수들을 초기화하고 Alloc 매소드를 호출한다.
    @param *pParameterContainer 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @param pLearningRate Optimizer의 learning rate
    @param momentum step size 조정 값
    @param pOptimizeDirection Optimizing의 방향(MAXIMIZE or MINIMIZE)
    @return 없음
    @see int Alloc(momentum)
    */
    NagOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float momentum, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "NagOptimizer::NagOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc(momentum);
    }

    /*!
    @brief NagOptimizer의 생성자.
    @details 맴버변수들을 초기화하고 Alloc 매소드를 호출한다.
    @param *pParameterContainer 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @param pLearningRate Optimizer의 learning rate
    @param m_momentum step size 조정 값
    @param weightDecayRate 가중치 매개변수가 클 때 패널티를 부과하는 값
    @param pOptimizeDirection Optimizing의 방향(MAXIMIZE or MINIMIZE)
    @return 없음
    @see int Alloc(momentum)
    */
    NagOptimizer(Container<Operator<DTYPE> *> *pParameterContainer, float pLearningRate, float momentum, float weightDecayRate, OptimizeDirection pOptimizeDirection) : Optimizer<DTYPE>(pParameterContainer, pLearningRate, weightDecayRate, pOptimizeDirection) {
        #ifdef __DEBUG__
        std::cout << "NagOptimizer::NagOptimizer(LossFunction<DTYPE> *, float, OptimizeDirection)" << '\n';
        #endif  // __DEBUG__
        m_ppParameter    = NULL;
        m_aaVelocity     = NULL;
        m_numOfParameter = 0;
        m_momentum       = 0.f;

        Alloc(momentum);
    }

    /*!
    @brief NagOptimizer의 소멸자
    @return 없음
    */
    ~NagOptimizer() {
        #ifdef __DEBUG__
        std::cout << "NagOptimizer::~NagOptimizer()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    /*!
    @brief Optimizer의 Delete 매소드
    @details 맴버 변수 m_aaVelocity의 메모리 할당을 해제한다.
    @return 없음
    */
    virtual void Delete(){
      if (m_aaVelocity) {
          delete m_aaVelocity;
          m_aaVelocity = NULL;
      }
    }

    /*!
    @brief Optimizer의 Alloc 매소드
    @details 맴버 변수 m_ppParameter, m_numOfParameter, m_aaVelocity를 초기화한다.
    @details m_aaVelocity를 m_ppParameter와 같은 Shape의 Tensor를 생성하여 넣는다.
    @param m_momentum step size 조정 값
    @return 성공 시 TRUE
    @see Container<Operator<DTYPE> *>* GetTrainableTensor()
    @see int GetTrainableTensorDegree()
    */
    int Alloc(float momentum) {
        m_ppParameter    = this->GetTrainableTensor();
        m_numOfParameter = this->GetTrainableTensorDegree();
        m_aaVelocity = new Container<Tensor<DTYPE> *>();

        Shape *pParameterGradShape = NULL;

        for (int i = 0; i < m_numOfParameter; i++) {
            pParameterGradShape = (*m_ppParameter)[i]->GetGradient()->GetShape();

            m_aaVelocity->Push(new Tensor<DTYPE>(new Shape(pParameterGradShape)));

            pParameterGradShape = NULL;
        }

        m_momentum = momentum;

        return TRUE;
    }

    /*!
    @brief 파라미터 값들을 업데이트 하는 메소드
    @details m_momentum 유무에 따라 UpdateParameter 호출과 에러 메세지 호출
    @return 성공 시 TRUE
    @see int UpdateParameter((Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity)
    */
    virtual int UpdateParameter() {
        if (m_momentum != 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                  UpdateParameter((*m_ppParameter)[i], (*m_aaVelocity)[i]);
            }
        }else{
            std::cout << "Don't execute UpdateParameter On CPU" << '\n';
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
    @details prev_Velocity 텐서 생성 후 현재 pVelocity 저장
    @details pVelocity를 Update 한다.
    @details m_momentum 값으로 조정된 prev_Velocity와 pVelocity의 연산으로 파라미터 Update
    @details 학습 초반 부, pFirstMomentum, pFirstVelocity는 0으로 biased 상태이므로 이를 unbiased 해주는 연산하여 업데이트 한다.
    @details signed_learning_rate와 pUnbiasedMomentum곱을 root가 적용된 pUnbiasedVelocity와 m_epsilon으로 나눈 값으로 weight(trainable_data)를 업데이트 한다.
    @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @param pVelocity 업데이트 할 pVelocity
    @return 성공 시 TURE
    */
    int UpdateParameter(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity) {
        Tensor<DTYPE> *trainable_data = pParameter->GetResult();
        Tensor<DTYPE> *gradient       = pParameter->GetGradient();
        Tensor<DTYPE> *prev_Velocity  = pVelocity;

        float learning_rate = this->GetOptimizeDirection() * this->GetLearningRate();

        int capacity = trainable_data->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*prev_Velocity)[i] = (*pVelocity)[i];
            (*pVelocity)[i]       = (m_momentum * (*pVelocity)[i]) + (learning_rate * (*gradient)[i]);
            (*trainable_data)[i] +=  -m_momentum * (*prev_Velocity)[i] + ((1.f + m_momentum) * (*pVelocity)[i]);
        }

        return TRUE;
    }

#ifdef __CUDNN__

    /*!
    @brief m_aaVelocity Tensor의 device를 idOfDevice번째 GPU로 바꾼다.
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
    @brief 파라미터 값들을 업데이트 하는 메소드(GPU)
    @details m_momentum 유무에 따라 UpdateParameterOnGPU 호출과 에러 메세지 호출
    @return 성공 시 TRUE
    @see int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity);
    */
    virtual int UpdateParameterOnGPU() {
        if (m_momentum != 0.f) {
            for (int i = 0; i < m_numOfParameter; i++) {
                UpdateParameterOnGPU((*m_ppParameter)[i], (*m_aaVelocity)[i]);
            }
        }else{
            std::cout << "Don't execute UpdataParameterOnGPU" << '\n';
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

    /*!
    @brief 파라미터 값들을 업데이트 하는 GPU 메서드.
    @details 실행시 해당 cu 파일의 __global__함수 실행.
    @param pParameter 업데이트 할 Tensor를 가지고 있는 Operator포인터
    @param pVelocity 업데이트 할 pVelocity
    @see __global__ void NagUpdate_kernel(float *pDevWeight, float *pDevAccGradient, int weightDim, float signed_learning_rate, float momentum, float weightDecayRate, float *pDevVelocity)
    @return 성공 시 TURE
    */
    int UpdateParameterOnGPU(Operator<DTYPE> *pParameter, Tensor<DTYPE> *pVelocity);


#endif  // if __CUDNN__
};


#endif  // GRADIENTDESCENTOPTIMIZER_H_
