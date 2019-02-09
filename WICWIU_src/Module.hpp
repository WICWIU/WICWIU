#ifndef __MODULE_H_
#define __MODULE_H_    value

#include "Operator_utils.hpp"

/*!
@class Module Operator들을 그래프로 구성해 모듈화하는 클래스
@details Operator들을 뉴럴 네트워크의 서브 그래프로 구성해 단일 Operator로서 할 수 없는 기능들을 수행하게 한다
@details Module은 하나의 Operator처럼 뉴럴 네트워크 안에서 작동한다
*/
template<typename DTYPE> class Module : public Operator<DTYPE>{
private:
    Container<Operator<DTYPE> *> *m_aaExcutableOperator;
    ///< Module을 구성하는 Operator들 중, 연산에 참여하는 Operator들의 포인터를 저장하는 Container 멤버 변수
    int m_numOfExcutableOperator;
    ///< Module을 구성하는 Operator들 중, 연산에 참여하는 Operator들의 개수

    Operator<DTYPE> *m_pLastOperator;
    ///< Module을 구성하는 Operator들 중, 순전파 순서 상 마지막에 해당하는 operator의 포인터

    Device m_Device;
    ///< 장치 사용 구분자, CPU 또는 GPU, Device 참고
    unsigned int m_idOfDevice;
    ///< GPU 사용 시, 사용하려는 GPU의 번호. CPU의 경우 -1

private:
    int  Alloc();
    void Delete();

public:
    Module(std::string pName = "No Name");
    virtual ~Module();

    Operator<DTYPE>                   * SetInput(Operator<DTYPE> *pInput);
    int                                 SetInput(int pNumOfInput, ...);

    int                                 IsInput(Operator<DTYPE> *pOperator);

    int                                 IsValid(Operator<DTYPE> *pOperator); // Graph 분석 시 node에 추가할 것인지 확인한다.

    Operator<DTYPE>                   * AnalyzeGraph(Operator<DTYPE> *pResultOperator);

    Container<Operator<DTYPE> *>      * GetExcutableOperatorContainer();
    int                                 GetNumOfExcutableOperator();

    virtual Tensor<DTYPE>             * GetResult() const;
    virtual Container<Tensor<DTYPE> *>* GetResultContainer();

    virtual Tensor<DTYPE>             * GetGradient() const;
    virtual Container<Tensor<DTYPE> *>* GetGradientContainer();

    virtual Tensor<DTYPE>             * GetDelta() const;
    virtual Container<Tensor<DTYPE> *>* GetDeltaContainer();

    int                                 SetModeTrain();
    int                                 SetModeAccumulate();
    int                                 SetModeInference();

    int                                 ForwardPropagate(int pTime = 0);
    int                                 BackPropagate(int pTime = 0);

    int                                 ResetResult();
    int                                 ResetGradient();

    // void                                PrintInformation();
    void                                PrintInformation(int level);

    void                                SetDeviceCPU();


    // int                                 SetResultOnCPU();
    // int                                 SetGradientOnCPU();
#ifdef __CUDNN__
    // int                                 SetResultOnGPU();
    // int                                 SetGradientOnGPU();

    // void SetDeviceGPU(unsigned int idOfDevice);
    void SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice);
    // void InitializeAttributeForGPU(unsigned int idOfDevice);

    int  ForwardPropagateOnGPU(int pTime = 0);
    int  BackPropagateOnGPU(int pTime = 0);
#endif  // if __CUDNN__
};


//////////////////////////////////////////////////////////////////////////////// for private method

/*!
@brief 동적으로 할당 받은 Module 클래스의 멤버 변수들을 할당 해제하는 메소드
@details 동적으로 할당 받은 Module 클래스의 Excutable Operator Container의 메모리를 할당 해제한다.
@return 없음
*/
template<typename DTYPE> int Module<DTYPE>::Alloc() {
    m_aaExcutableOperator = new Container<Operator<DTYPE> *>();
    return TRUE;
}

/*!
@brief 동적으로 할당 받은 Module 클래스의 멤버 변수들을 할당 해제하는 메소드
@details 동적으로 할당 받은 Module 클래스의 Excutable Operator Container의 메모리를 할당 해제한다.
@return 없음
*/
template<typename DTYPE> void Module<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "Module<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__

    if (m_aaExcutableOperator) {
        Operator<DTYPE> **OperatorContainer = m_aaExcutableOperator->GetRawData();

        for (int i = 0; i < m_numOfExcutableOperator; i++) {
            delete OperatorContainer[i];
            OperatorContainer[i] = NULL;
        }
        delete m_aaExcutableOperator;
        m_aaExcutableOperator = NULL;
    }
}

//////////////////////////////////////////////////////////////////////////////// for public method

/*!
@brief Module 클래스 생성자
@details 각 멤버 변수들을 초기화하고 Module 클래스를 생성한다.
@details 각 포인터들을 NULL 값으로, 각 정수 타입 변수들은 0으로 초기화하고 Module<DTYPE>::Alloc() 메소드를 호출한다.
@see Module<DTYPE>::Alloc()
@return 없음
*/
template<typename DTYPE> Module<DTYPE>::Module(std::string pName) : Operator<DTYPE>(pName) {
    #ifdef __DEBUG__
    std::cout << "Module<DTYPE>::Module()" << '\n';
    #endif  // __DEBUG__
    m_aaExcutableOperator    = NULL;
    m_numOfExcutableOperator = 0;
    m_pLastOperator          = NULL;
    m_idOfDevice             = -1;

    Alloc();
}

/*!
@brief Module 클래스 소멸자
@details 동적으로 할당 받은 Module 클래스의 멤버 변수들을 할당 해제하고 클래스를 소멸시킨다.
@return 없음
@see Module<DTYPE>::Delete()
*/
template<typename DTYPE> Module<DTYPE>::~Module() {
    #ifdef __DEBUG__
    std::cout << "Module<DTYPE>::~Module()" << '\n';
    #endif  // __DEBUG__

    this->Delete();
}

template<typename DTYPE> Operator<DTYPE> *Module<DTYPE>::SetInput(Operator<DTYPE> *pInput) {
    this->AddEdgebetweenOperators(pInput);

    return pInput;
}

template<typename DTYPE> int Module<DTYPE>::SetInput(int pNumOfInput, ...) {
    Operator<DTYPE> *temp = NULL;

    va_list ap;
    va_start(ap, pNumOfInput);

    for (int i = 0; i < pNumOfInput; i++) {
        temp = va_arg(ap, Operator<DTYPE> *);
        this->SetInput(temp);
    }

    va_end(ap);
    return TRUE;
}

/*!
@brief 해당 Operator가 Module의 Input인지 확인하는 메소드
@details 매개변수로 받은 Operator가 Module의 Input Container에 포함되어 있는 지 확인한다.
@param pOperator Input 여부를 확인하고자 하는 Operator
@return Input container에 포함되어 있는 경우 TRUE, 포함되어 있지 않는 경우 FALSE를 반환한다.
*/
template<typename DTYPE> int Module<DTYPE>::IsInput(Operator<DTYPE> *pOperator) {
    Container<Operator<DTYPE> *> *m_apInput = this->GetInputContainer();
    int m_InputDegree                       = m_apInput->GetSize();

    for (int i = 0; i < m_InputDegree; i++) {
        if ((*m_apInput)[i] == pOperator) return TRUE;
    }

    return FALSE;
}

/*!
@brief 해당 Operator의 Output Operator들이 모듈 그래프에 중복으로 포함되는 지 확인하는 메소드
@details 해당 Operator의 Output container 멤버 변수에 담겨 있는 Operator들이 Module의 Excutable Operator container에 중복되어 포함되어 있는 지 여부를 확인한다.
@param pOperator Output Container 멤버 변수가 Excutable Operator Container에 포함되어 있는 지 확인하고자 하는 Operator
@return 해당 Operator의 Output Container 멤버 변수가 Excutable Operator Container에 중복되어 포함되어 있으면 TRUE를 아니면 FALSE를 반환한다.
*/
template<typename DTYPE> int Module<DTYPE>::IsValid(Operator<DTYPE> *pOperator) {
    Container<Operator<DTYPE> *> *prevOp = pOperator->GetOutputContainer();
    int numOfOutputEdge                  = prevOp->GetSize();
    int check                            = 0;

    // every Output node is already in Excutable Operator
    for (int i = 0; i < numOfOutputEdge; i++) {
        for (int j = 0; j < m_numOfExcutableOperator; j++) {
            if ((*m_aaExcutableOperator)[j] == (*prevOp)[i]) {
                check++;
                break;
            }
        }

        if (check != (i + 1)) return FALSE;
    }

    return TRUE;
}

/*!
@brief 학습 가능한 형태로 모듈 그래프를 구성해주는 메소드
@details 모듈의 Output에 해당하는 Operator를 매개변수로 받아 너비 우선 탐색으로 모듈 그래프를 구성한다.
@details 매개변수로 받은 모듈의 Output에 해당하는 Operator를 시작으로 모듈의 Input에 해당하는 Operator까지 역순으로, Operator가 Input Tensor 및 학습 파라미터인 경우 Module 클래스의 Input Container 멤버 변수에 추가하고 나머지 경우에는 Module 클래스의 Excutable Operator Container 멤버 변수에 추가한다.
@details NeuralNetwork 클래스의 Excutable Operator Container 멤버 변수에 Operator들이 모두 추가되면 Container를 역순으로 변경한다.
@details Operator 탐색 순서는 너비 우선 탐색을 따르며, 매개변수로 받은 Output Operator부터 해당 Operator의 Input Operator 리스트를 너비 우선 탐색 방식을 이용해 순서대로 진행한다.
@details 각 Operator들은 Module<DTYPE>::IsValid(Operator<DTYPE> *pOperator) 메소드를 이용하여 모듈 그래프 안에서의 중복 여부를 확인하며 중복되는 경우 그래프에 추가하지 않는다.
@param pResultOperator 그래프를 구성하고자 하는 신경망의 Output에 해당하는 Operator
@return 매개변수로 받은 그래프를 구성하고자 하는 신경망의 Output에 해당하는 Operator
*/
template<typename DTYPE> Operator<DTYPE> *Module<DTYPE>::AnalyzeGraph(Operator<DTYPE> *pResultOperator) {
    // BFS
    Container<Operator<DTYPE> *> queue;

    queue.Push(pResultOperator);
    m_pLastOperator = pResultOperator;

    Container<Operator<DTYPE> *> *nextOp = NULL;
    Container<Operator<DTYPE> *> *prevOp = NULL;
    int numOfInputEdge                   = 0;

    Operator<DTYPE> *out = NULL;

    while (queue.GetSize() > 0) {
        out = queue.Pop();

        if (!(this->IsInput(out))) {
            if (this->IsValid(out)) {
                // std::cout << out->GetName() << '\n';

                if (out->GetIsTensorholder()) {
                    this->AddEdgebetweenOperators(out);
                } else {
                    m_aaExcutableOperator->Push(out);
                    m_numOfExcutableOperator++;
                }

                nextOp         = out->GetInputContainer();
                numOfInputEdge = nextOp->GetSize();

                for (int i = 0; i < numOfInputEdge; i++) {
                    prevOp = (*nextOp)[i]->GetOutputContainer();
                    prevOp->Pop(out);

                    queue.Push((*nextOp)[i]);
                }
            } else continue;
        } else continue;
    }
    // std::cout << '\n';

    m_aaExcutableOperator->Reverse();

    // cut output edge of input operator
    // Container<Operator<DTYPE> *> *m_apInput = this->GetInputContainer();
    // int m_InputDegree                       = m_apInput->GetSize();

    // for (int i = 0; i < m_InputDegree; i++) {
    // Operator<DTYPE> *pInput = (*m_apInput)[i];
    // Container<Operator<DTYPE> *> *prevOp = pInput->GetOutputContainer();
    // int numOfOutputEdge                  = prevOp->GetSize();
    //
    // for (int j = 0; j < numOfOutputEdge; j++) {
    // for (int k = 0; k < m_numOfExcutableOperator; k++) {
    // if ((*m_aaExcutableOperator)[j] == (*prevOp)[i]) {
    // (*prevOp)[i] = NULL;
    // }
    // }
    // }
    // }

    // std::cout << "m_aaExcutableOperator : " << '\n';
    //
    // for (int i = 0; i < m_numOfExcutableOperator; i++) {
    // std::cout << (*m_aaExcutableOperator)[i]->GetName() << '\n';
    // }
    // std::cout << '\n';
    //

    return pResultOperator;
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Module<DTYPE>::GetExcutableOperatorContainer() {
    return m_aaExcutableOperator;
}

template<typename DTYPE> int Module<DTYPE>::GetNumOfExcutableOperator() {
    return m_numOfExcutableOperator;
}

template<typename DTYPE> Tensor<DTYPE> *Module<DTYPE>::GetResult() const {
    return m_pLastOperator->GetResult();
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Module<DTYPE>::GetResultContainer() {
    return m_pLastOperator->GetResultContainer();
}

template<typename DTYPE> Tensor<DTYPE> *Module<DTYPE>::GetGradient() const {
    return m_pLastOperator->GetGradient();
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Module<DTYPE>::GetGradientContainer() {
    return m_pLastOperator->GetGradientContainer();
}

template<typename DTYPE> Tensor<DTYPE> *Module<DTYPE>::GetDelta() const {
    return m_pLastOperator->GetDelta();
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Module<DTYPE>::GetDeltaContainer() {
    return m_pLastOperator->GetDeltaContainer();
}

template<typename DTYPE> int Module<DTYPE>::SetModeTrain() {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->SetModeTrain();
    }
    return TRUE;
}

template<typename DTYPE> int Module<DTYPE>::SetModeAccumulate() {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->SetModeAccumulate();
    }
    return TRUE;
}

template<typename DTYPE> int Module<DTYPE>::SetModeInference() {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->SetModeInference();
    }
    return TRUE;
}

/*!
@brief 모듈 그래프의 순전파를 수행하는 메소드
@details Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::ForwardPropagate(int pTime) 메소드를 순서대로 호출한다.
@param pTime 각 ForwardPropagate 메소드에 전달할 Time의 인덱스
@return TRUE
*/
template<typename DTYPE> int Module<DTYPE>::ForwardPropagate(int pTime) {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->ForwardPropagate(pTime);
    }
    return TRUE;
}

/*!
@brief 모듈 그래프의 역전파를 수행하는 메소드
@details 역순으로 Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::ForwardPropagate(int pTime) 메소드를 호출한다.
@param pTime 각 ForwardPropagate 메소드에 전달할 Time의 인덱스
@return TRUE
*/
template<typename DTYPE> int Module<DTYPE>::BackPropagate(int pTime) {
    for (int i = m_numOfExcutableOperator - 1; i >= 0; i--) {
        (*m_aaExcutableOperator)[i]->BackPropagate(pTime);
    }
    return TRUE;
}

/*!
@brief 연산에 참여하는 Operator들의 Result Container를 초기화시킨다.
@details Excutable Operator Container에 포함되어 있는 각 Operator들에서 Operator<DTYPE>::ResetResult() 메소드를 호출한다.
@return TRUE
*/
template<typename DTYPE> int Module<DTYPE>::ResetResult() {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->ResetResult();
    }
    return TRUE;
}

/*!
@brief 연산에 참여하는 Operator들의 Gradient Container를 초기화시킨다.
@details Excutable Operator Container에 포함되어 있는 각 Operator들에서 Operator<DTYPE>::ResetGradient() 메소드를 호출한다.
@return TRUE
*/
template<typename DTYPE> int Module<DTYPE>::ResetGradient() {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->ResetGradient();
    }
    return TRUE;
}

// template<typename DTYPE> void Module<DTYPE>::PrintInformation() {
// std::cout << this->GetName() << " : ";
// std::cout << this->GetResult()->GetShape() << '\n';
//
// for (int i = 0; i < m_numOfExcutableOperator; i++) {
// std::cout << "-- ";
// (*m_aaExcutableOperator)[i]->PrintInformation();
// }
// }

template<typename DTYPE> void Module<DTYPE>::PrintInformation(int level) {
    for (int j = 0; j < level; j++) {
        std::cout << "-- ";
    }
    std::cout << this->GetName() << " : ";
    std::cout << this->GetResult()->GetShape() << '\n';

    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->PrintInformation(level + 1);
    }
}

/*!
@brief 모듈 그래프 학습에 사용되는 장치를 CPU로 전환하는 메소드
@details Module의 Device 멤버변수를 CPU로 전환하고, Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::SetDeviceCPU() 메소드를 순서대로 호출한다.
@return 없음
*/
template<typename DTYPE> void Module<DTYPE>::SetDeviceCPU() {
    this->SetDevice(CPU);

    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->SetDeviceCPU();
    }
}

#ifdef __CUDNN__

// template<typename DTYPE> void Module<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
// this->SetDevice(GPU);
//
// for (int i = 0; i < m_numOfExcutableOperator; i++) {
// (*m_aaExcutableOperator)[i]->SetDeviceGPU(idOfDevice);
// }
// }

/*!
@brief Module 클래스의 device 맴버 변수를 GPU로 변경한다.
@details LossFunction의 Result와 Gradient의 Device를 GPU로 변경한다.
@param pCudnnHandle cudnn 라이브러리를 가리키는 구조체 포인터.
@param idOfDevice 사용하고자 하는 GPU번호
*/
template<typename DTYPE> void Module<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) {
    checkCudaErrors(cudaSetDevice(idOfDevice));
    this->SetDevice(GPU);
    this->SetDeviceID(idOfDevice);
    this->SetCudnnHandle(pCudnnHandle);

    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->SetDeviceGPU(pCudnnHandle, idOfDevice);
    }
}

// template<typename DTYPE> void Module<DTYPE>::InitializeAttributeForGPU(unsigned int idOfDevice) {
// for (int i = 0; i < m_numOfExcutableOperator; i++) {
// (*m_aaExcutableOperator)[i]->InitializeAttributeForGPU(idOfDevice);
// }
// }

/*!
@brief GPU를 이용해 모듈 그래프의 순전파를 수행하는 메소드
@details Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::ForwardPropagateOnGPU(int pTime) 메소드를 순서대로 호출한다.
@param pTime 각 ForwardPropagateOnGPU 메소드에 전달할 Time의 인덱스
@return TRUE
*/
template<typename DTYPE> int Module<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    for (int i = 0; i < m_numOfExcutableOperator; i++) {
        (*m_aaExcutableOperator)[i]->ForwardPropagateOnGPU(pTime);
    }
    return TRUE;
}

/*!
@brief GPU를 이용해 모듈 그래프의 역전파를 수행하는 메소드
@details 역순으로 Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::BackPropagateOnGPU(int pTime) 메소드를 호출한다.
@param pTime 각 BackPropagateOnGPU 메소드에 전달할 Time의 인덱스
@return TRUE
*/
template<typename DTYPE> int Module<DTYPE>::BackPropagateOnGPU(int pTime) {
    for (int i = m_numOfExcutableOperator - 1; i >= 0; i--) {
        (*m_aaExcutableOperator)[i]->BackPropagateOnGPU(pTime);
    }
    return TRUE;
}

#endif  // if __CUDNN__

#endif  // ifndef __MODULE__
