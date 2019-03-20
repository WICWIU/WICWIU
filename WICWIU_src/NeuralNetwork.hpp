#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Optimizer_utils.hpp"

/*!
 * @class NeuralNetwork 뉴럴 네트워크 모델 생성, 학습 및 평가를 총괄하는 클래스
 * @details Operator, Module, Loss Function, Optimizer 클래스를 생성 및 활용해 뉴럴 네트워크를 구성하고 학습시킨다
 */
template<typename DTYPE> class NeuralNetwork : public Module<DTYPE>{
private:
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

    LossFunction<DTYPE>* SetLossFunction(LossFunction<DTYPE> *pLossFunction);
    Optimizer<DTYPE>   * SetOptimizer(Optimizer<DTYPE> *pOptimizer);

    Operator<DTYPE>    * GetResultOperator();
    Operator<DTYPE>    * GetResult();

    LossFunction<DTYPE>* GetLossFunction();

    Optimizer<DTYPE>   * GetOptimizer();


    void                 SetDeviceCPU();
    void                 SetDeviceCPUOnNeuralNetwork();

    int                  Train();
    int                  Test();

    int                  TrainOnCPU();
    int                  TestOnCPU();

    int                  TrainOnGPU();
    int                  TestOnGPU();

    float                GetAccuracy(int numOfClass = 10);
    int                  GetMaxIndex(Tensor<DTYPE> *data, int ba, int ti, int numOfClass);
    float                GetTop5Accuracy(int numOfClass);
    void                 GetTop5Index(Tensor<DTYPE> *data, int *top5Index, int ba, int ti, int numOfClass);
    float                GetLoss();

    void                 PrintGraphInformation();

    int                  ResetLossFunctionResult();
    int                  ResetLossFunctionGradient();

    int                  ResetParameterGradient();

    Operator<DTYPE>    * SearchOperator(std::string pName);

#ifdef __CUDNN__

    void SetDeviceGPU(unsigned int idOfDevice);
    void SetDeviceGPUOnNeuralNetwork(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice);
    int  SetDeviceID(unsigned int idOfDevice);
#endif  // __CUDNN__
};


//////////////////////////////////////////////////////////////////////////////// for private method

/*!
 * @brief NeuralNetwork 클래스의 Container 멤버 변수들을 동적으로 할당해주는 메소드
 * @details NeuralNetwork 클래스의 Operator, Excutable Operator, Input, Parameter Container들 각각에 대해 메모리를 동적으로 할당한다.
 * @return TRUE
 */
template<typename DTYPE> int NeuralNetwork<DTYPE>::Alloc() {
    return TRUE;
}

/*!
 * @brief 동적으로 할당 받은 NeuralNetwork 클래스의 멤버 변수들을 할당 해제하는 메소드
 * @details 동적으로 할당 받은 NeuralNetwork 클래스의 Operator, Excutable Operator, Input, Parameter Container들과 LossFunction, Optimizer의 메모리를 할당 해제한다.
 * @return 없음
 */
template<typename DTYPE> void NeuralNetwork<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "NeuralNetwork<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__
    // int size = 0;

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
 * @brief GPU 연산을 사용하기 위해 CUDNN Handler를 생성하는 메소드
 * @return 없음
 */
template<typename DTYPE> int NeuralNetwork<DTYPE>::AllocOnGPU() {
    // checkCudaErrors(cudaSetDevice(2));
    checkCUDNN(cudnnCreate(&m_cudnnHandle));
}

/*!
 * @brief GPU 연산을 사용하지 않기 위해 CUDNN Handler를 파괴하는 메소드
 * @return 없음
 */
template<typename DTYPE> void NeuralNetwork<DTYPE>::DeleteOnGPU() {
    //// checkCudaErrors(cudaDeviceSynchronize());
    //// checkCudaErrors(cudaDeviceSynchronize());
    if (m_cudnnHandle) checkCUDNN(cudnnDestroy(m_cudnnHandle));
}

#endif  // if __CUDNN__

//////////////////////////////////////////////////////////////////////////////// for public method


/*!
 * @brief NeuralNetwork 클래스 생성자
 * @details 각 멤버 변수들을 초기화하고 NeuralNetwork 클래스를 생성한다.
 * @details 각 포인터들을 NULL 값으로, 각 정수 타입 변수들은 0으로, Device는 CPU로 초기화하고 NeuralNetwork<DTYPE>::Alloc() 메소드를 호출한다.
 * @return 없음
 * @see NeuralNetwork<DTYPE>::Alloc()
 */
template<typename DTYPE> NeuralNetwork<DTYPE>::NeuralNetwork() : Module<DTYPE>() {
    #ifdef __DEBUG__
    std::cout << "NeuralNetwork<DTYPE>::NeuralNetwork()" << '\n';
    #endif  // __DEBUG__

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
 * @brief NeuralNetwork 클래스 소멸자
 * @details 동적으로 할당 받은 NeuralNetwork 클래스의 멤버 변수들을 할당 해제하고 클래스를 소멸시킨다.
 * @return 없음
 * @see NeuralNetwork<DTYPE>::Delete()
 */
template<typename DTYPE> NeuralNetwork<DTYPE>::~NeuralNetwork() {
    #ifdef __DEBUG__
    std::cout << "NeuralNetwork<DTYPE>::~NeuralNetwork()" << '\n';
    #endif  // __DEBUG__

    this->Delete();
}

/*!
 * @brief 특정 Loss Function을 매개 변수로 받아 이를 신경망의 Loss Function로 지정해주는 메소드
 * @param pLossFunction 신경망의 Loss Function로 지정하고자 하는 Loss Function
 * @return 매개변수로 받은 Loss Function
 */
template<typename DTYPE> LossFunction<DTYPE> *NeuralNetwork<DTYPE>::SetLossFunction(LossFunction<DTYPE> *pLossFunction) {
    m_aLossFunction = pLossFunction;
    return pLossFunction;
}

/*!
 * @brief 특정 Optimizer를 매개 변수로 받아 이를 신경망의 Optimizer로 지정해주는 메소드
 * @param pLossFunction 신경망의 Optimizer로 지정하고자 하는 Optimizer
 * @return 매개변수로 받은 Optimizer
 */
template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::SetOptimizer(Optimizer<DTYPE> *pOptimizer) {
    m_aOptimizer = pOptimizer;
    return pOptimizer;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::GetResultOperator() {
    return this->GetResult();
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::GetResult() {
    // return m_apExcutableOperator->GetLast();
    return this->GetExcutableOperatorContainer()->GetLast();
}

template<typename DTYPE> LossFunction<DTYPE> *NeuralNetwork<DTYPE>::GetLossFunction() {
    return m_aLossFunction;
}

template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::GetOptimizer() {
    return m_aOptimizer;
}

/*!
 * @brief 신경망 그래프 학습에 사용되는 장치를 CPU로 전환하는 메소드
 * @details NeuralNetwork의 Device 멤버변수를 CPU로 전환하고, Excutable Operator Container의 각 Operator들에서 Operator<DTYPE>::SetDeviceCPU() 메소드를 순서대로 호출하고, Lossfunction의 LossFunction<DTYPE>::SetDeviceCPU() 메소드를 호출한다.
 * @return 없음
 */
template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceCPU() {
    if (m_Device != CPU) this->SetDeviceCPUOnNeuralNetwork();
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceCPUOnNeuralNetwork() {
    m_Device = CPU;
    this->SetDeviceCPUOnModule();
    m_aLossFunction->SetDeviceCPU();
}

/*!
 * @brief 신경망의 학습을 진행하는 메소드
 * @details NeuralNetwork의 Device 멤버 변수를 확인하여 CPU 시 NeuralNetwork<DTYPE>::TrainingOnCPU()을 호출하고, GPU 시 NeuralNetwork<DTYPE>::TrainingOnGPU()을 호출한다.
 * @return 성공 시 TRUE, m_Device 멤버 변수가 잘못된 값을 갖고 있을 때 FALSE를 반환한다.
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
 * @brief 신경망의 테스트를 진행하는 메소드
 * @details NeuralNetwork의 Device 멤버 변수를 확인하여 CPU 시 NeuralNetwork<DTYPE>::TestingOnCPU()을 호출하고, GPU 시 NeuralNetwork<DTYPE>::TestingOnGPU()을 호출한다.
 * @return 성공 시 TRUE, m_Device 멤버 변수가 잘못된 값을 갖고 있을 때 FALSE를 반환한다.
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
 * @brief CPU를 활용해 신경망을 학습시키는 메소드
 * @details 순서대로 Excutable Operator들의 Result와 Gradient를 초기화하고 Loss Function의 Result와 Gradient를 초기화하고 ForwardPropagate, BackwardPropagate 메소드를 호출하고 Optimizer로 파라미터를 학습시킨다.
 * @details 각 메소드 참조
 * @return TRUE
 * @see NeuralNetwork<DTYPE>::ResetOperatorResult() NeuralNetwork<DTYPE>::ResetOperatorGradient() NeuralNetwork<DTYPE>::ResetLossFunctionResult() NeuralNetwork<DTYPE>::ResetLossFunctionGradient()
 * @see NeuralNetwork<DTYPE>::ForwardPropagate() NeuralNetwork<DTYPE>::BackPropagate() Optimizer<DTYPE>::UpdateParameter()
 */
template<typename DTYPE> int NeuralNetwork<DTYPE>::TrainOnCPU() {
    // this->ResetOperatorResult();
    // this->ResetOperatorGradient();
    this->ResetResult();
    this->ResetGradient();
    this->ResetLossFunctionResult();
    this->ResetLossFunctionGradient();

    this->ForwardPropagate();
    m_aLossFunction->ForwardPropagate();
    m_aLossFunction->BackPropagate();
    this->BackPropagate();

    m_aOptimizer->UpdateParameter();

    return TRUE;
}

/*!
 * @brief CPU를 활용해 신경망을 테스트하는 메소드
 * @details 순서대로 Excutable Operator들의 Result를 초기화하고 Loss Function의 Result를 초기화하고 ForwardPropagate메소드를 호출한다.
 * @details 각 메소드 참조
 * @return TRUE
 * @see NeuralNetwork<DTYPE>::ResetOperatorResult() NeuralNetwork<DTYPE>::ResetLossFunctionResult() NeuralNetwork<DTYPE>::ForwardPropagate()
 */
template<typename DTYPE> int NeuralNetwork<DTYPE>::TestOnCPU() {
    // this->ResetOperatorResult();
    this->ResetResult();
    this->ResetLossFunctionResult();

    this->ForwardPropagate();
    m_aLossFunction->ForwardPropagate();
    return TRUE;
}

/*!
 * @brief GPU를 활용해 신경망을 학습시키는 메소드
 * @details 순서대로 Excutable Operator들의 Result와 Gradient를 초기화하고 Loss Function의 Result와 Gradient를 초기화하고
 * @detaisl ForwardPropagateOnGPU, BackwardPropagateOnGPU 메소드를 호출하고 Optimizer로 파라미터를 학습시킨다.
 * @details 각 메소드 참조
 * @return TRUE
 * @see NeuralNetwork<DTYPE>::ResetOperatorResult() NeuralNetwork<DTYPE>::ResetOperatorGradient() NeuralNetwork<DTYPE>::ResetLossFunctionResult() NeuralNetwork<DTYPE>::ResetLossFunctionGradient()
 * @see NeuralNetwork<DTYPE>::ForwardPropagateOnGPU() NeuralNetwork<DTYPE>::BackPropagateOnGPU() Optimizer<DTYPE>::UpdateParameterOnGPU()
 */
template<typename DTYPE> int NeuralNetwork<DTYPE>::TrainOnGPU() {
#ifdef __CUDNN__
    // this->ResetOperatorResult();
    // this->ResetOperatorGradient();
    this->ResetResult();
    this->ResetGradient();
    this->ResetLossFunctionResult();
    this->ResetLossFunctionGradient();

    this->ForwardPropagateOnGPU();
    m_aLossFunction->ForwardPropagateOnGPU();
    m_aLossFunction->BackPropagateOnGPU();
    this->BackPropagateOnGPU();

    m_aOptimizer->UpdateParameterOnGPU();
#else  // __CUDNN__
    std::cout << "There is no GPU option!" << '\n';
    exit(-1);
#endif  // __CUDNN__

    return TRUE;
}

/*!
 * @brief GPU를 활용해 신경망을 테스트하는 메소드
 * @details 순서대로 Excutable Operator들의 Result를 초기화하고 Loss Function의 Result를 초기화하고 ForwardPropagateOnGPU메소드를 호출한다.
 * @details 각 메소드 참조
 * @return TRUE
 * @see NeuralNetwork<DTYPE>::ResetOperatorResult() NeuralNetwork<DTYPE>::ResetLossFunctionResult() NeuralNetwork<DTYPE>::ForwardPropagateOnGPU()
 */
template<typename DTYPE> int NeuralNetwork<DTYPE>::TestOnGPU() {
#ifdef __CUDNN__
    // this->ResetOperatorResult();
    this->ResetResult();
    this->ResetLossFunctionResult();

    this->ForwardPropagateOnGPU();
    m_aLossFunction->ForwardPropagateOnGPU();
#else  // __CUDNN__
    std::cout << "There is no GPU option!" << '\n';
    exit(-1);
#endif  // __CUDNN__

    return TRUE;
}

/*!
 * @brief 분류(Classification)를 위해 학습된 신경망의 Top 1 Accuracy를 계산하는 메소드
 * @param numOfClass 데이터의 분류(Classification)에 이용되는 label의 개수
 * @return 신경망의 Top 1 Accuracy : 0. ~ 1.
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
 * @brief Tensor의 LongArray의 Element들 중 가장 큰 값의 인덱스를 계산해 반환하는 메소드
 * @param data 탐색하고자 하는 Tensor
 * @param ba Tensor의 batch Size
 * @param ti Tensor의 Time Size
 * @param numOfClass Tensor의 LongArray의 Element 개수
 * @return 매개변수로 전달받은 Tensor의 LongArray의 Element들 중 가장 큰 값의 인덱스
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
 * @brief 분류(Classification)를 위해 학습된 신경망의 Top 5 Accuracy를 계산하는 메소드
 * @param numOfClass 데이터의 분류(Classification)에 이용되는 label의 개수
 * @return 신경망의 Accuracy : 0. ~ 1.
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
 * @brief Tensor의 LongArray의 Element들 중 가장 큰 다섯 개 값에 대한 인덱스를 계산해 반환하는 메소드
 * @param data 탐색하고자 하는 Tensor
 * @param ba Tensor의 batch Size
 * @param ti Tensor의 Time Size
 * @param numOfClass Tensor의 LongArray의 Element 개수
 * @return 매개변수로 전달받은 Tensor의 LongArray의 Element들 중 가장 큰 다섯 개 값에 대한 인덱스
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
 * @brief 데이터에 대해 학습된 신경망의 평균 Loss를 계산하여 반환하는 메소드
 * @return 학습된 신경망의 평균 Loss
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
 * @brief 신경망 그래프의 각 구성 요소에 대해 정보를 출력하는 메소드
 * @return 없음
 * @see Operator<DTYPE>::PrintInformation() LossFunction<DTYPE>::GetName()
 */
template<typename DTYPE> void NeuralNetwork<DTYPE>::PrintGraphInformation() {
    std::cout << "Graph Structure: " << "\n\n";

    for (int i = 0; i < this->GetNumOfExcutableOperator(); i++) {
        (*this->GetExcutableOperatorContainer())[i]->PrintInformation(0);
        std::cout << '\n';
    }

    std::cout << "LossFunction: " << m_aLossFunction->GetName() << '\n';
    // std::cout << "Optimizern: " << m_aOptimizer->GetName() << '\n';
}

/*!
 * @brief LossFunction의 Result Tensor를 초기화시킨다.
 * @details LossFunction의 LossFunction<DTYPE>::ResetResult() 메소드를 호출한다.
 * @return TRUE
 */
template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetLossFunctionResult() {
    m_aLossFunction->ResetResult();
    return TRUE;
}

/*!
 * @brief LossFunction의 Gradient Tensor를 초기화시킨다.
 * @details LossFunction의 Lossfunction<DTYPE>::ResetGradient() 메소드를 호출한다.
 * @return TRUE
 */
template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetLossFunctionGradient() {
    m_aLossFunction->ResetGradient();
    return TRUE;
}

/*!
 * @brief Optimizer의 Gradient와 Parameter들의 Gradient를 초기화시킨다.
 * @details Optimizer의 Optimzier<DTYPE>::ResetParameterGradient() 메소드를 호출한다.
 * @return TRUE
 */
template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetParameterGradient() {
    m_aOptimizer->ResetParameterGradient();
    return TRUE;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::SearchOperator(std::string pName) {
    std::string name = "NULL";

    // for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
    // name = (*m_apExcutableOperator)[i]->GetName();
    //
    // if (name == pName) return (*m_apExcutableOperator)[i];
    // }

    for (int i = 0; i < this->GetNumOfExcutableOperator(); i++) {
        name = (*this->GetExcutableOperatorContainer())[i]->GetName();

        if (name == pName) return (*this->GetExcutableOperatorContainer())[i];
    }

    return NULL;
}

#ifdef __CUDNN__

/*!
 * @brief 신경망 그래프 학습에 사용되는 장치를 GPU로 전환하는 메소드
 * @details 파라미터로 전달받은 GPU 장치 번호에 해당하는 GPU에 메모리를 할당하고 NeuralNetwork의 Device 멤버변수를 GPU로 전환한다
 * @details Excutable Operator Container, Parameter Operator Container, Input Operator Container의 각 Operator들에서 Operator<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) 메소드를 순서대로 호출한다
 * @details Lossfunction의 LossFunction<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) 메소드를 호출하고, Optimizer의 (cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) 메소드를 호출한다.
 * @param idOfDevice 학습에 이용하려는 GPU 장치 번호
 * @return 없음
 */
template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
    if (m_Device != GPU) {
        checkCudaErrors(cudaSetDevice(idOfDevice));
        m_Device = GPU;
        this->AllocOnGPU();
        this->SetDeviceGPUOnModule(m_cudnnHandle, idOfDevice);
        this->SetDeviceGPUOnNeuralNetwork(m_cudnnHandle, idOfDevice);
    }
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceGPUOnNeuralNetwork(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) {
    m_aLossFunction->SetDeviceGPU(m_cudnnHandle, idOfDevice);
    m_aOptimizer->SetDeviceGPU(m_cudnnHandle, idOfDevice);
}

/*!
 * @brief 파라미터로 입력받은 값으로 GPU 장치 번호를 변경한다.
 * @param idOfDevice 해당 GPU 장치 번호
 * @return TRUE
 */
template<typename DTYPE> int NeuralNetwork<DTYPE>::SetDeviceID(unsigned int idOfDevice) {
    m_idOfDevice = idOfDevice;
    return TRUE;
}

#endif  // __CUDNN__

#endif  // NEURALNETWORK_H_
