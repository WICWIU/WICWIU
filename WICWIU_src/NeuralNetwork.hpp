#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Optimizer_utils.hpp"

template<typename DTYPE> class NeuralNetwork {
private:
    Container<Operator<DTYPE> *> *m_aaOperator;
    Container<Operator<DTYPE> *> *m_apExcutableOperator;
    Container<Operator<DTYPE> *> *m_apInput;
    Container<Operator<DTYPE> *> *m_apParameter;

    int m_Operatordegree;
    int m_ExcutableOperatorDegree;
    int m_InputDegree;
    int m_ParameterDegree;

    LossFunction<DTYPE> *m_aLossFunction;
    Optimizer<DTYPE> *m_aOptimizer;

    Device m_Device;
    int m_idOfDevice = -1;  // 추후 수정

#ifdef __CUDNN__
    cudnnHandle_t m_cudnnHandle;
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

    Operator<DTYPE>             * SerchOperator(std::string pName);

    int                           Save(FILE *fileForSave);
    int                           Load(FILE *fileForLoad);

#ifdef __CUDNN__
    int                           ForwardPropagateOnGPU(int pTime = 0);
    int                           BackPropagateOnGPU(int pTime = 0);

    void                          SetDeviceGPU(unsigned int idOfDevice);
    int                           SetDeviceID(unsigned int idOfDevice);
#endif  // __CUDNN__
};


//////////////////////////////////////////////////////////////////////////////// for private method

template<typename DTYPE> int NeuralNetwork<DTYPE>::Alloc() {
    m_aaOperator          = new Container<Operator<DTYPE> *>();
    m_apExcutableOperator = new Container<Operator<DTYPE> *>();
    m_apInput             = new Container<Operator<DTYPE> *>();
    m_apParameter         = new Container<Operator<DTYPE> *>();
    return TRUE;
}

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
template<typename DTYPE> int NeuralNetwork<DTYPE>::AllocOnGPU() {
    // checkCudaErrors(cudaSetDevice(2));
    checkCUDNN(cudnnCreate(&m_cudnnHandle));
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::DeleteOnGPU() {
    // // checkCudaErrors(cudaDeviceSynchronize());
    // // checkCudaErrors(cudaDeviceSynchronize());
    if (m_cudnnHandle) checkCUDNN(cudnnDestroy(m_cudnnHandle));
}

#endif  // if __CUDNN__

//////////////////////////////////////////////////////////////////////////////// for public method


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

    m_Device = CPU;

#ifdef __CUDNN__
    m_cudnnHandle = NULL;
#endif  // if __CUDNN__

    Alloc();
}

template<typename DTYPE> NeuralNetwork<DTYPE>::~NeuralNetwork() {
    #ifdef __DEBUG__
    std::cout << "NeuralNetwork<DTYPE>::~NeuralNetwork()" << '\n';
    #endif  // __DEBUG__

    this->Delete();
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::SetInput(Operator<DTYPE> *pInput) {
    m_aaOperator->Push(pInput);
    m_Operatordegree++;

    m_apInput->Push(pInput);
    m_InputDegree++;
    return pInput;
}

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

template<typename DTYPE> int NeuralNetwork<DTYPE>::IsInput(Operator<DTYPE> *pOperator) {
    for (int i = 0; i < m_InputDegree; i++) {
        if ((*m_apInput)[i] == pOperator) return TRUE;
    }

    return FALSE;
}

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

template<typename DTYPE> LossFunction<DTYPE> *NeuralNetwork<DTYPE>::SetLossFunction(LossFunction<DTYPE> *pLossFunction) {
    m_aLossFunction = pLossFunction;
    return pLossFunction;
}

template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::SetOptimizer(Optimizer<DTYPE> *pOptimizer) {
    m_aOptimizer = pOptimizer;
    return pOptimizer;
}

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

template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagate(int pTime) {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->ForwardPropagate(pTime);
    }
    m_aLossFunction->ForwardPropagate(pTime);

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::BackPropagate(int pTime) {
    m_aLossFunction->BackPropagate(pTime);

    for (int i = m_ExcutableOperatorDegree - 1; i >= 0; i--) {
        (*m_apExcutableOperator)[i]->BackPropagate(pTime);
    }
    return TRUE;
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceCPU() {
    m_Device = CPU;

    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetDeviceCPU();
    }
    m_aLossFunction->SetDeviceCPU();
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeTrain() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetModeTrain();
    }
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeAccumulate() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetModeAccumulate();
    }
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeInference() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetModeInference();
    }
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Train() {
    if (m_Device == CPU) {
        this->TrainOnCPU();
    } else if (m_Device == GPU) {
        this->TrainOnGPU();
    } else return FALSE;

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Test() {
    if (m_Device == CPU) {
        this->TestOnCPU();
    } else if (m_Device == GPU) {
        this->TestOnGPU();
    } else return FALSE;

    return TRUE;
}

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

template<typename DTYPE> int NeuralNetwork<DTYPE>::TestOnCPU() {
    this->ResetOperatorResult();
    this->ResetLossFunctionResult();

    this->ForwardPropagate();
    return TRUE;
}

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
            int pred_index[5] = { 0, }; // for Initialize
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

template<typename DTYPE> void NeuralNetwork<DTYPE>::PrintGraphInformation() {
    std::cout << "Graph Structure: " << "\n\n";

    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->PrintInformation(0);
        std::cout << '\n';
    }

    std::cout << "LossFunction: " << m_aLossFunction->GetName() << '\n';
    // std::cout << "Optimizern: " << m_aOptimizer->GetName() << '\n';
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetOperatorResult() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->ResetResult();
    }
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetOperatorGradient() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->ResetGradient();
    }
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetLossFunctionResult() {
    m_aLossFunction->ResetResult();
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetLossFunctionGradient() {
    m_aLossFunction->ResetGradient();
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetParameterGradient() {
    m_aOptimizer->ResetParameterGradient();
    return TRUE;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::SerchOperator(std::string pName) {
    std::string name = "NULL";

    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        name = (*m_apExcutableOperator)[i]->GetName();

        if (name == pName) return (*m_apExcutableOperator)[i];
    }

    return NULL;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Save(FILE *fileForSave) {
    for (int i = 0; i < m_ParameterDegree; i++) {
        // important order
        (*m_apParameter)[i]->Save(fileForSave);
    }
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Load(FILE *fileForLoad) {
    for (int i = 0; i < m_ParameterDegree; i++) {
        // important order
        (*m_apParameter)[i]->Load(fileForLoad);
    }
    return TRUE;
}

#ifdef __CUDNN__
template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->ForwardPropagateOnGPU(pTime);
    }
    m_aLossFunction->ForwardPropagateOnGPU(pTime);

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::BackPropagateOnGPU(int pTime) {
    m_aLossFunction->BackPropagateOnGPU(pTime);

    for (int i = m_ExcutableOperatorDegree - 1; i >= 0; i--) {
        (*m_apExcutableOperator)[i]->BackPropagateOnGPU(pTime);
    }
    return TRUE;
}

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

template<typename DTYPE> int NeuralNetwork<DTYPE>::SetDeviceID(unsigned int idOfDevice) {
    m_idOfDevice = idOfDevice;
    return TRUE;
}

#endif  // __CUDNN__

#endif  // NEURALNETWORK_H_
