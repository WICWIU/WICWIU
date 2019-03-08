#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Optimizer_utils.hpp"

/*!
@class NeuralNetwork 뉴럴 네트워크 모델 생성, 학습 및 평가를 총괄하는 클래스
@details Operator, Module, Loss Function, Optimizer 클래스를 생성 및 활용해 뉴럴 네트워크를 구성하고 학습시킨다
*/
template<typename DTYPE> class NeuralNetwork {
private:
    Container<Operator<DTYPE> *> *m_aaOperator;
    ///< 신경망의 전체 Operator들의 포인터를 담고 있는 Container의 포인터 멤버 변수  //// Excutable + Input + Parameter
    Container<Operator<DTYPE> *> *m_apExcutableOperator;
    ///< 순전파 시 연산을 수행하는 Operator들의 포인터를 담고 있는 Container의 포인터 멤버 변수
    Container<Operator<DTYPE> *> *m_apInput;
    ///< 신경망의 최초 Input이 되는 Operator들의 포인터를 담고 있는 Container의 포인터 멤버 변수
    Container<Operator<DTYPE> *> *m_apParameter;
    ///< 신경망의 학습이 가능한 파라미터에 해당하는 Operator들의 포인터를 담고 있는 Container의 포인터 멤버 변수

    int m_Operatordegree;
    ///< 해당 클래스의 Operator Container 멤버 변수의 Element의 개수
    int m_ExcutableOperatorDegree;
    ///< 해당 클래스의 Excutable Operator Container 멤버 변수의 Element의 개수
    int m_InputDegree;
    ///< 해당 클래스의 Input Container 멤버 변수의 Element의 개수
    int m_ParameterDegree;
    ///< 해당 클래스의 Parameter Container 멤버 변수의 Element의 개수

    LossFunction<DTYPE> *m_aLossFunction;
    ///< 신경망의 손실함수에 해당하는 LossFunction의 포인터 멤버 변수
    Optimizer<DTYPE> *m_aOptimizer;
    ///< 신경망의 Optimizer에 해당하는 Optimizer의 포인터 멤버 변수

    Device m_Device;
    ///< 장치 사용 구분자, CPU 또는 GPU, Device 참고
    int m_idOfDevice;
     ///< GPU 사용 시, 사용하려는 GPU의 번호. CPU의 경우 -1

#ifdef __CUDNN__
    cudnnHandle_t m_cudnnHandle;
    ///< cudnn handler
#endif  // if __CUDNN__

private:
    int  Alloc();
    void Delete();

#ifdef __CUDNN__
    int  AllocOnGPU();
    void DeleteOnGPU();
#endif  // if __CUDNN__

public:
    NeuralNetwork();
    virtual ~NeuralNetwork();

    Operator<DTYPE>             * SetInput(Operator<DTYPE> *pInput);
    int                           SetInput(int pNumOfInput, ...);
    int                           IsInput(Operator<DTYPE> *pOperator);

    int                           IsValid(Operator<DTYPE> *pOperator); // Graph 분석 시 node에 추가할 것인지 확인한다.

    Operator<DTYPE>             * AnalyzeGraph(Operator<DTYPE> *pResultOperator);
    LossFunction<DTYPE>         * SetLossFunction(LossFunction<DTYPE> *pLossFunction);
    Optimizer<DTYPE>            * SetOptimizer(Optimizer<DTYPE> *pOptimizer);
    int                           FeedInputTensor(int pNumOfInput, ...);
    // =======

    Container<Operator<DTYPE> *>* GetInputContainer();

    Operator<DTYPE>             * GetResultOperator();
    Operator<DTYPE>             * GetResult();

    Container<Operator<DTYPE> *>* GetExcutableOperatorContainer();

    Container<Operator<DTYPE> *>* GetParameterContainer();
    Container<Operator<DTYPE> *>* GetParameter();

    LossFunction<DTYPE>         * GetLossFunction();

    Optimizer<DTYPE>            * GetOptimizer();

    int                           ForwardPropagate(int pTime = 0);
    int                           BackPropagate(int pTime = 0);

    void                          SetDeviceCPU();

    void                          SetModeTrain();
    void                          SetModeAccumulate();
    void                          SetModeInference();

    int                           Train();
    int                           Test();

    int                           TrainOnCPU();
    int                           TestOnCPU();

    int                           TrainOnGPU();
    int                           TestOnGPU();

    float                         GetAccuracy(int numOfClass = 10);
    int                           GetMaxIndex(Tensor<DTYPE> *data, int ba, int ti, int numOfClass);
    float                         GetTop5Accuracy(int numOfClass);
    void                          GetTop5Index(Tensor<DTYPE> *data, int *top5Index, int ba, int ti, int numOfClass);
    float                         GetLoss();

    void                          PrintGraphInformation();

    int                           ResetOperatorResult();
    int                           ResetOperatorGradient();

    int                           ResetLossFunctionResult();
    int                           ResetLossFunctionGradient();

    int                           ResetParameterGradient();

    Operator<DTYPE>             * SearchOperator(std::string pName);

    int                           Save();
    int                           Load();

#ifdef __CUDNN__
    int                           ForwardPropagateOnGPU(int pTime = 0);
    int                           BackPropagateOnGPU(int pTime = 0);

    void                          SetDeviceGPU(unsigned int idOfDevice);
    int                           SetDeviceID(unsigned int idOfDevice);
#endif  // __CUDNN__
};


//////////////////////////////////////////////////////////////////////////////// for private method

/*!
@brief NeuralNetwork 클래스의 Container 멤버 변수들을 동적으로 할당해주는 메소드
@details NeuralNetwork 클래스의 Operator, Excutable Operator, Input, Parameter Container들 각각에 대해 메모리를 동적으로 할당한다.
@return TRUE
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::Alloc() {
    m_aaOperator          = new Container<Operator<DTYPE> *>();
    m_apExcutableOperator = new Container<Operator<DTYPE> *>();
    m_apInput             = new Container<Operator<DTYPE> *>();
    m_apParameter         = new Container<Operator<DTYPE> *>();
    return TRUE;
}

/*!
@brief 동적으로 할당 받은 NeuralNetwork 클래스의 멤버 변수들을 할당 해제하는 메소드
@details 동적으로 할당 받은 NeuralNetwork 클래스의 Operator, Excutable Operator, Input, Parameter Container들과 LossFunction, Optimizer의 메모리를 할당 해제한다.
@return 없음
*/
template<typename DTYPE> void NeuralNetwork<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "NeuralNetwork<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__
    int size = 0;

    if (m_aaOperator) {
        size = m_aaOperator->GetSize();
        Operator<DTYPE> **OperatorContainer = m_aaOperator->GetRawData();

        for (int i = 0; i < size; i++) {
            if ((*m_aaOperator)[i]) {
                delete OperatorContainer[i];
                OperatorContainer[i] = NULL;
            }
        }
        delete m_aaOperator;
        m_aaOperator = NULL;
    }

    if (m_apExcutableOperator) {
        delete m_apExcutableOperator;
        m_apExcutableOperator = NULL;
    }

    if (m_apInput) {
        delete m_apInput;
        m_apInput = NULL;
    }

    if (m_apParameter) {
        delete m_apParameter;
        m_apParameter = NULL;
    }

    if (m_aLossFunction) {
        delete m_aLossFunction;
        m_aLossFunction = NULL;
    }

    if (m_aOptimizer) {
        delete m_aOptimizer;
        m_aOptimizer = NULL;
    }

#ifdef __CUDNN__
    this->DeleteOnGPU();
#endif  // if __CUDNN__
}

#ifdef __CUDNN__
/*!
@brief GPU 연산을 사용하기 위해 CUDNN Handler를 생성하는 메소드
@return 없음
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::AllocOnGPU() {
    // checkCudaErrors(cudaSetDevice(2));
    checkCUDNN(cudnnCreate(&m_cudnnHandle));
}

/*!
@brief GPU 연산을 사용하지 않기 위해 CUDNN Handler를 파괴하는 메소드
@return 없음
*/
template<typename DTYPE> void NeuralNetwork<DTYPE>::DeleteOnGPU() {
    //// checkCudaErrors(cudaDeviceSynchronize());
    //// checkCudaErrors(cudaDeviceSynchronize());
    if (m_cudnnHandle) checkCUDNN(cudnnDestroy(m_cudnnHandle));
}

#endif  // if __CUDNN__

//////////////////////////////////////////////////////////////////////////////// for public method


/*!
@brief NeuralNetwork 클래스 생성자
@details 각 멤버 변수들을 초기화하고 NeuralNetwork 클래스를 생성한다.
@details 각 포인터들을 NULL 값으로, 각 정수 타입 변수들은 0으로, Device는 CPU로 초기화하고 NeuralNetwork<DTYPE>::Alloc() 메소드를 호출한다.
@return 없음
@see NeuralNetwork<DTYPE>::Alloc()
*/
template<typename DTYPE> NeuralNetwork<DTYPE>::NeuralNetwork() {
    #ifdef __DEBUG__
    std::cout << "NeuralNetwork<DTYPE>::NeuralNetwork()" << '\n';
    #endif  // __DEBUG__

    m_aaOperator          = NULL;
    m_apExcutableOperator = NULL;
    m_apInput             = NULL;
    m_apParameter         = NULL;

    m_Operatordegree          = 0;
    m_ExcutableOperatorDegree = 0;
    m_InputDegree             = 0;
    m_ParameterDegree         = 0;

    m_aLossFunction = NULL;
    m_aOptimizer    = NULL;

    m_Device     = CPU;
    m_idOfDevice = -1;

#ifdef __CUDNN__
    m_cudnnHandle = NULL;
#endif  // if __CUDNN__

    Alloc();
}

/*!
@brief NeuralNetwork 클래스 소멸자
@details 동적으로 할당 받은 NeuralNetwork 클래스의 멤버 변수들을 할당 해제하고 클래스를 소멸시킨다.
@return 없음
@see NeuralNetwork<DTYPE>::Delete()
*/
template<typename DTYPE> NeuralNetwork<DTYPE>::~NeuralNetwork() {
    #ifdef __DEBUG__
    std::cout << "NeuralNetwork<DTYPE>::~NeuralNetwork()" << '\n';
    #endif  // __DEBUG__

    this->Delete();
}

/*!
@brief Operator를 신경망의 Input에 추가하는 메소드
@details 매개 변수로 받은 Operator를 NeuralNetwork 클래스의 Operator, Input Container에 추가하고 각 degree를 1만큼 증가시킨다.
@param pInput Input으로 추가하고자 하는 Operator
@return 매개변수로 받은 Operator
*/
template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::SetInput(Operator<DTYPE> *pInput) {
    m_aaOperator->Push(pInput);
    m_Operatordegree++;

    m_apInput->Push(pInput);
    m_InputDegree++;
    return pInput;
}

/*!
@brief Operator 리스트를 신경망의 Input에 추가하는 메소드
@details Operator 개수와 Operator 리스트를 매개변수로 받아서, 각각의 Operator에 대해서 NeuralNetwork<DTYPE>::SetInput(Operator<DTYPE> *pInput)를 호출한다.
@param pNumOfInput Input에 추가하고자 하는 Operator의 개수
@param ... Input에 추가하고자 하는 Operator의 리스트
@return TRUE
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::SetInput(int pNumOfInput, ...) {
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
@brief 해당 Operator가 신경망의 Input인지 확인하는 메소드
@details 매개변수로 받은 Operator가 NeuralNetwork의 Input Container에 포함되어 있는 지 확인한다.
@param pOperator Input 여부를 확인하고자 하는 Operator
@return Input container에 포함되어 있는 경우 TRUE, 포함되어 있지 않는 경우 FALSE를 반환한다.
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::IsInput(Operator<DTYPE> *pOperator) {
    for (int i = 0; i < m_InputDegree; i++) {
        if ((*m_apInput)[i] == pOperator) return TRUE;
    }

    return FALSE;
}

/*!
@brief 해당 Operator의 Output Operator들이 신경망 그래프에 중복으로 포함되는 지 확인하는 메소드
@details 해당 Operator의 Output container 멤버 변수에 담겨 있는 Operator들이 NeuralNetwork의 Excutable Operator container에 중복되어 포함되어 있는 지 여부를 확인한다.
@param pOperator Output Container 멤버 변수가 Excutable Operator Container에 포함되어 있는 지 확인하고자 하는 Operator
@return 해당 Operator의 Output Container 멤버 변수가 Excutable Operator Container에 중복되어 포함되어 있으면 TRUE를 아니면 FALSE를 반환한다.
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::IsValid(Operator<DTYPE> *pOperator) {
    Container<Operator<DTYPE> *> *prevOp = pOperator->GetOutputContainer();
    int numOfOutputEdge                  = prevOp->GetSize();
    int check                            = 0;

    // every Output node is already in Excutable Operator
    for (int i = 0; i < numOfOutputEdge; i++) {
        for (int j = 0; j < m_ExcutableOperatorDegree; j++) {
            if ((*m_apExcutableOperator)[j] == (*prevOp)[i]) {
                check++;
                break;
            }
        }

        if (check != (i + 1)) return FALSE;
    }

    return TRUE;
}

/*!
@brief 학습 가능한 형태로 신경망 그래프를 구성해주는 메소드
@details 신경망의 Output에 해당하는 Operator를 매개변수로 받아 너비 우선 탐색으로 신경망 그래프를 구성한다.
@details 매개변수로 받은 신경망의 Output에 해당하는 Operator를 시작으로 신경망의 Input에 해당하는 Operator까지 역순으로 NeuralNetwork 클래스의 Container 멤버 변수들에 Operator들을 추가한다.
@details NeuralNetwork 클래스의 Container 멤버 변수들에 Operator들을 모두 추가한 후, 각 Container들을 역순으로 변경한다.
@details Operator 탐색 순서는 너비 우선 탐색을 따르며, 매개변수로 받은 Output Operator부터 해당 Operator의 Input Operator 리스트를 너비 우선 탐색 방식을 이용해 순서대로 진행한다.
@details 신경망의 각 Operator들은 Operator Container에 순서대로 추가되며, 연산에 참여하는 Operator의 경우 Excutable Conatainer에 학습 파라미터에 해당하는 Operator의 경우 Parameter Container에 순서대로 추가된다.
@details 각 Operator들은 NeuralNetwork<DTYPE>::IsValid(Operator<DTYPE> *pOperator) 메소드를 이용하여 신경망 그래프 안에서의 중복 여부를 확인하며 중복되는 경우 그래프에 추가하지 않는다.
@param pResultOperator 그래프를 구성하고자 하는 신경망의 Output에 해당하는 Operator
@return 매개변수로 받은 그래프를 구성하고자 하는 신경망의 Output에 해당하는 Operator
*/
template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AnalyzeGraph(Operator<DTYPE> *pResultOperator) {
    // BFS
    Container<Operator<DTYPE> *> queue;
    queue.Push(pResultOperator);
    Operator<DTYPE> *out                 = NULL;
    Container<Operator<DTYPE> *> *nextOp = NULL;
    int numOfInputEdge                   = 0;

    while (queue.GetSize() > 0) {
        out = queue.Pop();

        if (!(this->IsInput(out))) {
            if (this->IsValid(out)) {
                // std::cout << out->GetName() << '\n';

                m_aaOperator->Push(out);
                m_Operatordegree++;

                if (out->GetIsTensorholder()) {
                    m_apParameter->Push(out);
                    m_ParameterDegree++;
                } else {
                    m_apExcutableOperator->Push(out);
                    m_ExcutableOperatorDegree++;
                }

                nextOp         = out->GetInputContainer();
                numOfInputEdge = nextOp->GetSize();

                // std::cout << numOfInputEdge << '\n';

                for (int i = 0; i < numOfInputEdge; i++) {
                    queue.Push((*nextOp)[i]);
                }
            } else continue;
        } else continue;
    }
    // std::cout << '\n';

    m_aaOperator->Reverse();
    m_apExcutableOperator->Reverse();
    m_apParameter->Reverse();

    // std::cout << "m_aaOperator : " << '\n';
    //
    // for (int i = 0; i < m_Operatordegree; i++) {
    // std::cout << (*m_aaOperator)[i]->GetName() << '\n';
    // }
    // std::cout << '\n';
    //
    // std::cout << "m_apExcutableOperator : " << '\n';
    //
    // for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
    // std::cout << (*m_apExcutableOperator)[i]->GetName() << '\n';
    // }
    // std::cout << '\n';
    //
    // std::cout << "m_apInput : " << '\n';
    //
    // for (int i = 0; i < m_InputDegree; i++) {
    // std::cout << (*m_apInput)[i]->GetName() << '\n';
    // }
    // std::cout << '\n';
    //
    // std::cout << "m_apParameter : " << '\n';
    //
    // for (int i = 0; i < m_ParameterDegree; i++) {
    // std::cout << (*m_apParameter)[i]->GetName() << '\n';
    // }
    // std::cout << '\n';

    return pResultOperator;
}

/*!
@brief 특정 Loss Function을 매개 변수로 받아 이를 신경망의 Loss Function로 지정해주는 메소드
@param pLossFunction 신경망의 Loss Function로 지정하고자 하는 Loss Function
@return 매개변수로 받은 Loss Function
*/
template<typename DTYPE> LossFunction<DTYPE> *NeuralNetwork<DTYPE>::SetLossFunction(LossFunction<DTYPE> *pLossFunction) {
    m_aLossFunction = pLossFunction;
    return pLossFunction;
}

/*!
@brief 특정 Optimizer를 매개 변수로 받아 이를 신경망의 Optimizer로 지정해주는 메소드
@param pLossFunction 신경망의 Optimizer로 지정하고자 하는 Optimizer
@return 매개변수로 받은 Optimizer
*/
template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::SetOptimizer(Optimizer<DTYPE> *pOptimizer) {
    m_aOptimizer = pOptimizer;
    return pOptimizer;
}

/*!
@brief 신경망에 Input 리스트를 추가하는 메소드
@details 매개변수로 받은 Tensor들을 순서대로 NeuralNetwork의 Input Container에 담겨 있는 Operator들의 Result로 설정한다.
@param pNumOfInput Input Container에 추가하고 싶은 Tensor들의 개수
@param ... Input Container에 추가하고 싶은 Tensor들의 리스트
@return TRUE
@see Operator<DTYPE>::SetResult(Tensor<DTYPE> *pTensor)
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::FeedInputTensor(int pNumOfInput, ...) {
    Tensor<DTYPE> *temp = NULL;

    va_list ap;
    va_start(ap, pNumOfInput);

    for (int i = 0; i < pNumOfInput; i++) {
        temp = va_arg(ap, Tensor<DTYPE> *);
        (*m_apInput)[i]->SetResult(temp);
    }

    va_end(ap);
    return TRUE;
}

template<typename DTYPE> Container<Operator<DTYPE> *> *NeuralNetwork<DTYPE>::GetInputContainer() {
    return m_apInput;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::GetResultOperator() {
    return this->GetResult();
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::GetResult() {
    return m_apExcutableOperator->GetLast();
}

template<typename DTYPE> Container<Operator<DTYPE> *> *NeuralNetwork<DTYPE>::GetExcutableOperatorContainer() {
    return m_apExcutableOperator;
}

template<typename DTYPE> Container<Operator<DTYPE> *> *NeuralNetwork<DTYPE>::GetParameterContainer() {
    return m_apParameter;
}

template<typename DTYPE> Container<Operator<DTYPE> *> *NeuralNetwork<DTYPE>::GetParameter() {
    return m_apParameter;
}

template<typename DTYPE> LossFunction<DTYPE> *NeuralNetwork<DTYPE>::GetLossFunction() {
    return m_aLossFunction;
}

template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::GetOptimizer() {
    return m_aOptimizer;
}

/*!
@brief 신경망 그래프의 순전파를 수행하는 메소드
@details Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::ForwardPropagate(int pTime) 메소드를 순서대로 호출하고, Lossfunction의 LossFunction<DTYPE>::ForwardPropagate(int pTime) 메소드를 호출한다.
@param pTime 각 ForwardPropagate 메소드에 전달할 Time의 인덱스
@return TRUE
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagate(int pTime) {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->ForwardPropagate(pTime);
    }
    m_aLossFunction->ForwardPropagate(pTime);

    return TRUE;
}

/*!
@brief 신경망 그래프의 역전파를 수행하는 메소드
@details Lossfunction의 LossFunction<DTYPE>::ForwardPropagate(int pTime) 메소드를 호출하고, Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::ForwardPropagate(int pTime) 메소드를 역순으로 호출한다.
@param pTime 각 ForwardPropagate 메소드에 전달할 Time의 인덱스
@return TRUE
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::BackPropagate(int pTime) {
    m_aLossFunction->BackPropagate(pTime);

    for (int i = m_ExcutableOperatorDegree - 1; i >= 0; i--) {
        (*m_apExcutableOperator)[i]->BackPropagate(pTime);
    }
    return TRUE;
}

/*!
@brief 신경망 그래프 학습에 사용되는 장치를 CPU로 전환하는 메소드
@details NeuralNetwork의 Device 멤버변수를 CPU로 전환하고, Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::SetDeviceCPU() 메소드를 순서대로 호출하고, Lossfunction의 LossFunction<DTYPE>::SetDeviceCPU() 메소드를 호출한다.
@return 없음
*/
template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceCPU() {
    m_Device = CPU;

    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetDeviceCPU();
    }
    m_aLossFunction->SetDeviceCPU();
}

/*!
@brief 신경망 그래프의 학습 모드를 TRAINING 상태로 전환하는 메소드
@details Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::SetModeTraining() 메소드를 순서대로 호출한다.
@return 없음
*/
template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeTrain() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetModeTrain();
    }
}

/*!
@brief 신경망 그래프의 학습 모드를 ACCUMULATING(Batch Normalization을 이용한 학습 시 사용) 상태로 전환하는 메소드
@details Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::SetModeAccumulating() 메소드를 순서대로 호출한다.
@return 없음
*/
template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeAccumulate() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetModeAccumulate();
    }
}

/*!
@brief 신경망 그래프의 학습 모드를 INFERENCING(테스트) 상태로 전환하는 메소드
@details Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::SetModeInferencing() 메소드를 순서대로 호출한다.
@return 없음
*/
template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeInference() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetModeInference();
    }
}

/*!
@brief 신경망의 학습을 진행하는 메소드
@details NeuralNetwork의 Device 멤버 변수를 확인하여 CPU 시 NeuralNetwork<DTYPE>::TrainingOnCPU()을 호출하고, GPU 시 NeuralNetwork<DTYPE>::TrainingOnGPU()을 호출한다.
@return 성공 시 TRUE, m_Device 멤버 변수가 잘못된 값을 갖고 있을 때 FALSE를 반환한다.
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::Train() {
    if (m_Device == CPU) {
        this->TrainOnCPU();
    } else if (m_Device == GPU) {
        this->TrainOnGPU();
    } else return FALSE;

    return TRUE;
}

/*!
@brief 신경망의 테스트를 진행하는 메소드
@details NeuralNetwork의 Device 멤버 변수를 확인하여 CPU 시 NeuralNetwork<DTYPE>::TestingOnCPU()을 호출하고, GPU 시 NeuralNetwork<DTYPE>::TestingOnGPU()을 호출한다.
@return 성공 시 TRUE, m_Device 멤버 변수가 잘못된 값을 갖고 있을 때 FALSE를 반환한다.
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::Test() {
    if (m_Device == CPU) {
        this->TestOnCPU();
    } else if (m_Device == GPU) {
        this->TestOnGPU();
    } else return FALSE;

    return TRUE;
}

/*!
@brief CPU를 활용해 신경망을 학습시키는 메소드
@details 순서대로 Excutable Operator들의 Result와 Gradient를 초기화하고 Loss Function의 Result와 Gradient를 초기화하고 ForwardPropagate, BackwardPropagate 메소드를 호출하고 Optimizer로 파라미터를 학습시킨다.
@details 각 메소드 참조
@return TRUE
@see NeuralNetwork<DTYPE>::ResetOperatorResult() NeuralNetwork<DTYPE>::ResetOperatorGradient() NeuralNetwork<DTYPE>::ResetLossFunctionResult() NeuralNetwork<DTYPE>::ResetLossFunctionGradient()
@see NeuralNetwork<DTYPE>::ForwardPropagate() NeuralNetwork<DTYPE>::BackPropagate() Optimizer<DTYPE>::UpdateParameter()
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::TrainOnCPU() {
    this->ResetOperatorResult();
    this->ResetOperatorGradient();
    this->ResetLossFunctionResult();
    this->ResetLossFunctionGradient();

    this->ForwardPropagate();
    this->BackPropagate();

    m_aOptimizer->UpdateParameter();

    return TRUE;
}

/*!
@brief CPU를 활용해 신경망을 테스트하는 메소드
@details 순서대로 Excutable Operator들의 Result를 초기화하고 Loss Function의 Result를 초기화하고 ForwardPropagate메소드를 호출한다.
@details 각 메소드 참조
@return TRUE
@see NeuralNetwork<DTYPE>::ResetOperatorResult() NeuralNetwork<DTYPE>::ResetLossFunctionResult() NeuralNetwork<DTYPE>::ForwardPropagate()
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::TestOnCPU() {
    this->ResetOperatorResult();
    this->ResetLossFunctionResult();

    this->ForwardPropagate();
    return TRUE;
}

/*!
@brief GPU를 활용해 신경망을 학습시키는 메소드
@details 순서대로 Excutable Operator들의 Result와 Gradient를 초기화하고 Loss Function의 Result와 Gradient를 초기화하고
@detaisl ForwardPropagateOnGPU, BackwardPropagateOnGPU 메소드를 호출하고 Optimizer로 파라미터를 학습시킨다.
@details 각 메소드 참조
@return TRUE
@see NeuralNetwork<DTYPE>::ResetOperatorResult() NeuralNetwork<DTYPE>::ResetOperatorGradient() NeuralNetwork<DTYPE>::ResetLossFunctionResult() NeuralNetwork<DTYPE>::ResetLossFunctionGradient()
@see NeuralNetwork<DTYPE>::ForwardPropagateOnGPU() NeuralNetwork<DTYPE>::BackPropagateOnGPU() Optimizer<DTYPE>::UpdateParameterOnGPU()
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::TrainOnGPU() {
#ifdef __CUDNN__
    this->ResetOperatorResult();
    this->ResetOperatorGradient();
    this->ResetLossFunctionResult();
    this->ResetLossFunctionGradient();

    this->ForwardPropagateOnGPU();
    this->BackPropagateOnGPU();

    m_aOptimizer->UpdateParameterOnGPU();
#else  // __CUDNN__
    std::cout << "There is no GPU option!" << '\n';
    exit(-1);
#endif  // __CUDNN__

    return TRUE;
}

/*!
@brief GPU를 활용해 신경망을 테스트하는 메소드
@details 순서대로 Excutable Operator들의 Result를 초기화하고 Loss Function의 Result를 초기화하고 ForwardPropagateOnGPU메소드를 호출한다.
@details 각 메소드 참조
@return TRUE
@see NeuralNetwork<DTYPE>::ResetOperatorResult() NeuralNetwork<DTYPE>::ResetLossFunctionResult() NeuralNetwork<DTYPE>::ForwardPropagateOnGPU()
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::TestOnGPU() {
#ifdef __CUDNN__
    this->ResetOperatorResult();
    this->ResetLossFunctionResult();

    this->ForwardPropagateOnGPU();
#else  // __CUDNN__
    std::cout << "There is no GPU option!" << '\n';
    exit(-1);
#endif  // __CUDNN__

    return TRUE;
}

/*!
@brief 분류(Classification)를 위해 학습된 신경망의 Top 1 Accuracy를 계산하는 메소드
@param numOfClass 데이터의 분류(Classification)에 이용되는 label의 개수
@return 신경망의 Top 1 Accuracy : 0. ~ 1.
*/
template<typename DTYPE> float NeuralNetwork<DTYPE>::GetAccuracy(int numOfClass) {
    Operator<DTYPE> *result = GetResultOperator();
    Operator<DTYPE> *label  = m_aLossFunction->GetLabel();

    int batchsize = label->GetResult()->GetBatchSize();
    int timesize  = label->GetResult()->GetTimeSize();

    Tensor<DTYPE> *pred = result->GetResult();
    Tensor<DTYPE> *ans  = label->GetResult();

    float accuracy = 0.f;

    int pred_index = 0;
    int ans_index  = 0;
    // printf("\n\n");

    for (int ba = 0; ba < batchsize; ba++) {
        for (int ti = 0; ti < timesize; ti++) {
            pred_index = GetMaxIndex(pred, ba, ti, numOfClass);
            ans_index  = GetMaxIndex(ans, ba, ti, numOfClass);
            // printf("%d, ", ans_index);

            if (pred_index == ans_index) {
                accuracy += 1.f;
            }
        }
    }
    // printf("\n\n");

    // return (float)((accuracy / 1) / 1);
    return (float)((accuracy / timesize) / batchsize);
}

/*!
@brief Tensor의 LongArray의 Element들 중 가장 큰 값의 인덱스를 계산해 반환하는 메소드
@param data 탐색하고자 하는 Tensor
@param ba Tensor의 batch Size
@param ti Tensor의 Time Size
@param numOfClass Tensor의 LongArray의 Element 개수
@return 매개변수로 전달받은 Tensor의 LongArray의 Element들 중 가장 큰 값의 인덱스
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::GetMaxIndex(Tensor<DTYPE> *data, int ba, int ti, int numOfClass) {
    Shape *pShape = data->GetShape();
    int    start  = Index5D(pShape, ti, ba, 0, 0, 0);
    int    end    = start + numOfClass;

    // Initial max value is first element
    DTYPE max       = (*data)[start];
    int   max_index = 0;

    for (int dim = start + 1; dim < end; dim++) {
        if ((*data)[dim] > max) {
            max       = (*data)[dim];
            max_index = dim - start;
        }
    }

    return max_index;
}

///////////////////////////////////////////
/*!
@brief 분류(Classification)를 위해 학습된 신경망의 Top 5 Accuracy를 계산하는 메소드
@param numOfClass 데이터의 분류(Classification)에 이용되는 label의 개수
@return 신경망의 Accuracy : 0. ~ 1.
*/
template<typename DTYPE> float NeuralNetwork<DTYPE>::GetTop5Accuracy(int numOfClass) {
    Operator<DTYPE> *result = GetResultOperator();
    Operator<DTYPE> *label  = m_aLossFunction->GetLabel();

    int batchsize = label->GetResult()->GetBatchSize();
    int timesize  = label->GetResult()->GetTimeSize();

    Tensor<DTYPE> *pred = result->GetResult();
    Tensor<DTYPE> *ans  = label->GetResult();

    float top5Accuracy = 0.f;

    for (int ba = 0; ba < batchsize; ba++) {
        for (int ti = 0; ti < timesize; ti++) {
            int pred_index[5] = { 0, };  // for Initialize
            int ans_index     = 0;

            GetTop5Index(pred, pred_index, ba, ti, numOfClass);
            ans_index = GetMaxIndex(ans, ba, ti, numOfClass);
            // printf("%d, ", ans_index);

            // pred_index[5] (top5Index) 중 하나라도 레이블과 같은 경우, 1을 더하고 break
            for (int i = 0; i < 5; i++) {
                // printf("pred_index[%d] = %d, ans_Index = %d\n", i, pred_index[i], ans_index);
                if (pred_index[i] == ans_index) {
                    top5Accuracy += 1.f;
                }
            }
        }
    }
    // printf("\n\n");

    // return (float)((top5Accuracy / 1) / 1);
    return (float)((top5Accuracy / timesize) / batchsize);
}

/*
 * 상위 5개 노드의 값과 인덱스를 위한 5칸짜리 어레이 두 개를 생성
 * Value, Index Array 각각 0으로 초기화
 *
 * 어레이의 4부터 0까지 순서대로 큰 값들을 저장,
 * 4인 경우 가장 큰 값과 인덱스, 0인 경우 5번째로 큰 값과 인덱스
 *
 * Index 어레이는 Accuracy 메소드에서 생성한 후, 포인터를 통해 전달
 * 텐서의 아웃풋 노드들과 하나씩 비교 및 스왑하면서 어레이를 채워감
 *
 * swap method의 경우 std::swap 이용
 * 각각의 아웃풋 노드들에 대해 먼저 0번째 값과 비교한 후,
 * 노드의 값이 더 큰 경우 0번째 값과 인덱스의 해당 노드의 값과 인덱스을 대입
 * 그 뒤 어레이의 원소들을 차례대로 비교하고 스왑이 필요한 경우 스왑, 필요 없는 경우 break (Sorting)
 */

/*!
@brief Tensor의 LongArray의 Element들 중 가장 큰 다섯 개 값에 대한 인덱스를 계산해 반환하는 메소드
@param data 탐색하고자 하는 Tensor
@param ba Tensor의 batch Size
@param ti Tensor의 Time Size
@param numOfClass Tensor의 LongArray의 Element 개수
@return 매개변수로 전달받은 Tensor의 LongArray의 Element들 중 가장 큰 다섯 개 값에 대한 인덱스
*/
template<typename DTYPE> void NeuralNetwork<DTYPE>::GetTop5Index(Tensor<DTYPE> *data, int *top5Index, int ba, int ti, int numOfClass) {
    Shape *pShape = data->GetShape();
    int    start  = Index5D(pShape, ti, ba, 0, 0, 0);
    int    end    = start + numOfClass;

    // Initialize array with 0
    DTYPE top5Value[5] = { 0, };

    // Find 5 top elements
    for (int dim = start; dim < end; dim++) {
        // printf("(*data)(%d) = %f, top5Value[0] = %f\n", dim, (float)(*data)[dim], (float)top5Value[0]);

        if ((*data)[dim] > top5Value[0]) {
            // printf("if((*data)[dim] > top5Value[0]) clear\n");
            top5Value[0] = (*data)[dim];
            top5Index[0] = dim - start;

            // printf("top5Value[0] = %f, top5Index[0] = %d\n", (float)top5Value[0], (float)top5index[0]);
            for (int i = 0; i < 4; i++) {
                // printf("for(int i = 0; i < 4; i++) clear\n");
                // printf("top5Value[0] = %f, top5Index[0] = %d\n", (float)top5Value[0], (float)top5index[0]);
                if (top5Value[i] > top5Value[i + 1]) {
                    // printf("if(top5Value[i] > top5Value[i+1]) clear\n");
                    // printf("top5Value[%d] = %f, top5Index[%d] = %d\n", i, (float)top5Value[i], i, (float)top5index[i]);
                    std::swap(top5Value[i], top5Value[i + 1]);
                    std::swap(top5Index[i], top5Index[i + 1]);
                    // printf("swap clear\n");
                    // printf("top5Value[%d] = %f, top5Index[%d] = %d\n", i, (float)top5Value[i], i, (float)top5index[i]);
                } else break;
            }
        }
    }
}

/*!
@brief 데이터에 대해 학습된 신경망의 평균 Loss를 계산하여 반환하는 메소드
@return 학습된 신경망의 평균 Loss
*/
template<typename DTYPE> float NeuralNetwork<DTYPE>::GetLoss() {
    float avg_loss = 0.f;

    int batchsize = m_aLossFunction->GetResult()->GetBatchSize();
    int timesize  = m_aLossFunction->GetResult()->GetTimeSize();

    for (int ba = 0; ba < batchsize; ba++) {
        for (int ti = 0; ti < timesize; ti++) {
            avg_loss += (*m_aLossFunction)[ba] / batchsize / timesize;
        }
    }

    return avg_loss;
}

/*!
@brief 신경망 그래프의 각 구성 요소에 대해 정보를 출력하는 메소드
@return 없음
@see Operator<DTYPE>::PrintInformation() LossFunction<DTYPE>::GetName()
*/
template<typename DTYPE> void NeuralNetwork<DTYPE>::PrintGraphInformation() {
    std::cout << "Graph Structure: " << "\n\n";

    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->PrintInformation(0);
        std::cout << '\n';
    }

    std::cout << "LossFunction: " << m_aLossFunction->GetName() << '\n';
    // std::cout << "Optimizern: " << m_aOptimizer->GetName() << '\n';
}

/*!
@brief 연산에 참여하는 Operator들의 Result Container를 초기화시킨다.
@details Excutable Operator Container에 포함되어 있는 각 Operator들에서 Operator<DTYPE>::ResetResult() 메소드를 호출한다.
@return TRUE
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetOperatorResult() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->ResetResult();
    }
    return TRUE;
}

/*!
@brief 연산에 참여하는 Operator들의 Gradient Container를 초기화시킨다.
@details Excutable Operator Container에 포함되어 있는 각 Operator들에서 Operator<DTYPE>::ResetGradient() 메소드를 호출한다.
@return TRUE
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetOperatorGradient() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->ResetGradient();
    }
    return TRUE;
}

/*!
@brief LossFunction의 Result Tensor를 초기화시킨다.
@details LossFunction의 LossFunction<DTYPE>::ResetResult() 메소드를 호출한다.
@return TRUE
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetLossFunctionResult() {
    m_aLossFunction->ResetResult();
    return TRUE;
}

/*!
@brief LossFunction의 Gradient Tensor를 초기화시킨다.
@details LossFunction의 Lossfunction<DTYPE>::ResetGradient() 메소드를 호출한다.
@return TRUE
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetLossFunctionGradient() {
    m_aLossFunction->ResetGradient();
    return TRUE;
}

/*!
@brief Optimizer의 Gradient와 Parameter들의 Gradient를 초기화시킨다.
@details Optimizer의 Optimzier<DTYPE>::ResetParameterGradient() 메소드를 호출한다.
@return TRUE
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetParameterGradient() {
    m_aOptimizer->ResetParameterGradient();
    return TRUE;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::SearchOperator(std::string pName) {
    std::string name = "NULL";

    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        name = (*m_apExcutableOperator)[i]->GetName();

        if (name == pName) return (*m_apExcutableOperator)[i];
    }

    return NULL;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Save() {
    for (int i = 0; i < m_ParameterDegree; i++) {
        // important order
        (*m_apParameter)[i]->Save(i);
    }
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Load() {
    for (int i = 0; i < m_ParameterDegree; i++) {
        // important order
        (*m_apParameter)[i]->Load(i);
    }
    return TRUE;
}

#ifdef __CUDNN__
/*!
@brief GPU를 활용해 신경망 그래프의 순전파를 수행하는 메소드
@details Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::ForwardPropagateOnGPU(int pTime) 메소드를 순서대로 호출하고, Lossfunction의 LossFunction<DTYPE>::ForwardPropagateOnGPU(int pTime) 메소드를 호출한다.
@param pTime 각 ForwardPropagateOnGPU 메소드에 전달할 Time의 인덱스
@return TRUE
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->ForwardPropagateOnGPU(pTime);
    }
    m_aLossFunction->ForwardPropagateOnGPU(pTime);

    return TRUE;
}

/*!
@brief GPU를 활용해 신경망 그래프의 역전파를 수행하는 메소드
@details Lossfunction의 LossFunction<DTYPE>::ForwardPropagateOnGPU(int pTime) 메소드를 호출하고, Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::ForwardPropagateOnGPU(int pTime) 메소드를 역순으로 호출한다.
@param pTime 각 ForwardPropagateOnGPU 메소드에 전달할 Time의 인덱스
@return TRUE
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::BackPropagateOnGPU(int pTime) {
    m_aLossFunction->BackPropagateOnGPU(pTime);

    for (int i = m_ExcutableOperatorDegree - 1; i >= 0; i--) {
        (*m_apExcutableOperator)[i]->BackPropagateOnGPU(pTime);
    }
    return TRUE;
}

/*!
@brief 신경망 그래프 학습에 사용되는 장치를 GPU로 전환하는 메소드
@details 파라미터로 전달받은 GPU 장치 번호에 해당하는 GPU에 메모리를 할당하고 NeuralNetwork의 Device 멤버변수를 GPU로 전환한다
@details Excutable Operator Container, Parameter Operator Container, Input Operator Container의 각 Operator들에서 Operator<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) 메소드를 순서대로 호출한다
@details Lossfunction의 LossFunction<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) 메소드를 호출하고, Optimizer의 (cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) 메소드를 호출한다.
@param idOfDevice 학습에 이용하려는 GPU 장치 번호
@return 없음
*/
template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
    // std::cout << "NeuralNetwork<DTYPE>::SetModeGPU()" << '\n';
    checkCudaErrors(cudaSetDevice(idOfDevice));

    m_Device = GPU;
    this->AllocOnGPU();

    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        // important order
        (*m_apExcutableOperator)[i]->SetDeviceGPU(m_cudnnHandle, idOfDevice);
    }

    for (int i = 0; i < m_ParameterDegree; i++) {
        // important order
        (*m_apParameter)[i]->SetDeviceGPU(m_cudnnHandle, idOfDevice);
    }

    for (int i = 0; i < m_InputDegree; i++) {
        // important order
        (*m_apInput)[i]->SetDeviceGPU(m_cudnnHandle, idOfDevice);
    }

    m_aLossFunction->SetDeviceGPU(m_cudnnHandle, idOfDevice);

    m_aOptimizer->SetDeviceGPU(m_cudnnHandle, idOfDevice);
}

/*!
@brief 파라미터로 입력받은 값으로 GPU 장치 번호를 변경한다.
@param idOfDevice 해당 GPU 장치 번호
@return TRUE
*/
template<typename DTYPE> int NeuralNetwork<DTYPE>::SetDeviceID(unsigned int idOfDevice) {
    m_idOfDevice = idOfDevice;
    return TRUE;
}

#endif  // __CUDNN__

#endif  // NEURALNETWORK_H_
