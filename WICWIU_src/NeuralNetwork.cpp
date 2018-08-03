#include "NeuralNetwork.h"

template class NeuralNetwork<int>;
template class NeuralNetwork<float>;
template class NeuralNetwork<double>;

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
    checkCudaErrors(cudaThreadSynchronize());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCUDNN(cudnnDestroy(m_cudnnHandle));
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

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeTraining() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetModeTraining();
    }
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeAccumulating() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetModeAccumulating();
    }
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeInferencing() {
    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        (*m_apExcutableOperator)[i]->SetModeInferencing();
    }
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Training() {
    if (m_Device == CPU) {
        this->TrainingOnCPU();
    } else if (m_Device == GPU) {
        this->TrainingOnGPU();
    } else return FALSE;

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Testing() {
    if (m_Device == CPU) {
        this->TestingOnCPU();
    } else if (m_Device == GPU) {
        this->TestingOnGPU();
    } else return FALSE;

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::TrainingOnCPU() {
    this->ResetOperatorResult();
    this->ResetOperatorGradient();
    this->ResetLossFunctionResult();
    this->ResetLossFunctionGradient();

    this->ForwardPropagate();
    this->BackPropagate();

    m_aOptimizer->UpdateParameter();

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::TestingOnCPU() {
    this->ResetOperatorResult();
    this->ResetLossFunctionResult();

    this->ForwardPropagate();
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::TrainingOnGPU() {
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

template<typename DTYPE> int NeuralNetwork<DTYPE>::TestingOnGPU() {
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

    for (int ba = 0; ba < batchsize; ba++) {
        for (int ti = 0; ti < timesize; ti++) {
            pred_index = GetMaxIndex(pred, ba, ti, numOfClass);
            ans_index  = GetMaxIndex(ans, ba, ti, numOfClass);

            if (pred_index == ans_index) {
                accuracy += 1.f;
            }
        }
    }

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
        (*m_apExcutableOperator)[i]->PrintInformation();
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

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceGPU() {
    // std::cout << "NeuralNetwork<DTYPE>::SetModeGPU()" << '\n';
    m_Device = GPU;
    this->AllocOnGPU();

    for (int i = 0; i < m_ExcutableOperatorDegree; i++) {
        // important order
        (*m_apExcutableOperator)[i]->SetDeviceGPU(m_cudnnHandle);
    }

    for (int i = 0; i < m_ParameterDegree; i++) {
        // important order
        (*m_apParameter)[i]->SetDeviceGPU(m_cudnnHandle);
    }
    m_aLossFunction->SetDeviceGPU(m_cudnnHandle);

    m_aOptimizer->SetCudnnHandle(m_cudnnHandle);
}

#endif  // __CUDNN__
