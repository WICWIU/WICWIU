#include "NeuralNetwork.h"

template class NeuralNetwork<int>;
template class NeuralNetwork<float>;
template class NeuralNetwork<double>;

template<typename DTYPE> NeuralNetwork<DTYPE>::NeuralNetwork() {
    std::cout << "NeuralNetwork<DTYPE>::NeuralNetwork()" << '\n';

    m_aaOperator     = NULL;
    m_aaTensorholder = NULL;
    m_aaLayer        = NULL;

    m_OperatorDegree     = 0;
    m_TensorholderDegree = 0;

    m_aLossFunction = NULL;
    m_aOptimizer = NULL;

    m_Device      = CPU;
    m_numOfThread = 1;

    Alloc();
}

template<typename DTYPE> NeuralNetwork<DTYPE>::~NeuralNetwork() {
    std::cout << "NeuralNetwork<DTYPE>::~NeuralNetwork()" << '\n';

    this->Delete();
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Alloc() {
    m_aaOperator     = new Container<Operator<DTYPE> *>();
    m_aaTensorholder = new Container<Tensorholder<DTYPE> *>();
    m_aaLayer        = new Container<Layer<DTYPE> *>();

#if __CUDNN__
    checkCUDNN(cudnnCreate(&m_cudnnHandle));
#endif  // if __CUDNN__

    return TRUE;
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::Delete() {
    std::cout << "NeuralNetwork<DTYPE>::Delete()" << '\n';
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

    if (m_aaTensorholder) {
        size = m_aaTensorholder->GetSize();
        Tensorholder<DTYPE> **TensorholderContainer = m_aaTensorholder->GetRawData();

        for (int i = 0; i < size; i++) {
            if ((*m_aaTensorholder)[i]) {
                delete TensorholderContainer[i];
                TensorholderContainer[i] = NULL;
            }
        }
        delete m_aaTensorholder;
        m_aaTensorholder = NULL;
    }

    if (m_aaLayer) {
        size = m_aaLayer->GetSize();
        Layer<DTYPE> **LayerContainer = m_aaLayer->GetRawData();

        for (int i = 0; i < size; i++) {
            if ((*m_aaLayer)[i]) {
                delete LayerContainer[i];
                LayerContainer[i] = NULL;
            }
        }
        delete m_aaLayer;
        m_aaLayer = NULL;
    }

    if (m_aLossFunction) {
        delete m_aLossFunction;
        m_aLossFunction = NULL;
    }

    if (m_aOptimizer) {
        delete m_aOptimizer;
        m_aOptimizer = NULL;
    }

#if __CUDNN__
    checkCudaErrors(cudaDeviceSynchronize());
    checkCUDNN(cudnnDestroy(m_cudnnHandle));
#endif  // if __CUDNN__
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AddOperator(Operator<DTYPE> *pOperator) {
    int pNumOfParameter = pOperator->GetNumOfParameter();

    m_aaOperator->Push(pOperator);
    m_OperatorDegree++;

    for (int i = 0; i < pNumOfParameter; i++) {
        m_aaTensorholder->Push(pOperator->PopParameter());
        m_TensorholderDegree++;
    }

    return pOperator;
}

template<typename DTYPE> Tensorholder<DTYPE> *NeuralNetwork<DTYPE>::AddTensorholder(Tensorholder<DTYPE> *pTensorholder) {
    m_aaTensorholder->Push(pTensorholder);
    m_TensorholderDegree++;
    return pTensorholder;
}

template<typename DTYPE> Tensorholder<DTYPE> *NeuralNetwork<DTYPE>::AddParameter(Tensorholder<DTYPE> *pTensorholder) {
    m_aaTensorholder->Push(pTensorholder);
    m_TensorholderDegree++;
    return pTensorholder;
}

// template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::AddLayer(Layer<DTYPE> *pLayer) {
// int pNumOfParameter = pLayer->GetNumOfParameter();
//
// m_aaOperator->Push(pLayer);
// m_OperatorDegree++;
//
// for (int i = 0; i < pNumOfParameter; i++) {
// m_aaTensorholder->Push(pLayer->PopParameter());
// m_TensorholderDegree++;
// }
//
// return pLayer;
// }

template<typename DTYPE> LossFunction<DTYPE> *NeuralNetwork<DTYPE>::SetLossFunction(LossFunction<DTYPE> *pLossFunction) {
    m_aLossFunction = pLossFunction;
    return pLossFunction;
}

template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::SetOptimizer(Optimizer<DTYPE> *pOptimizer) {
    m_aOptimizer = pOptimizer;
    return pOptimizer;
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::GetResultOperator() {
    return m_aaOperator->GetLast();
}

template<typename DTYPE> Operator<DTYPE> *NeuralNetwork<DTYPE>::GetResult() {
    return m_aaOperator->GetLast();
}

template<typename DTYPE> Container<Operator<DTYPE> *> *NeuralNetwork<DTYPE>::GetOperatorContainer() {
    return m_aaOperator;
}

template<typename DTYPE> Container<Tensorholder<DTYPE> *> *NeuralNetwork<DTYPE>::GetTensorholder() {
    return m_aaTensorholder;
}

template<typename DTYPE> Container<Tensorholder<DTYPE> *> *NeuralNetwork<DTYPE>::GetParameter() {
    return m_aaTensorholder;
}

template<typename DTYPE> LossFunction<DTYPE> *NeuralNetwork<DTYPE>::GetLossFunction() {
    return m_aLossFunction;
}

template<typename DTYPE> Optimizer<DTYPE> *NeuralNetwork<DTYPE>::GetOptimizer() {
    return m_aOptimizer;
}

template<typename DTYPE> float NeuralNetwork<DTYPE>::GetAccuracy() {
    Operator<DTYPE> *result = GetResultOperator();
    Operator<DTYPE> *label  = m_aLossFunction->GetLabel();

    int batch = label->GetResult()->GetBatchSize();

    Tensor<DTYPE> *pred = result->GetResult();
    Tensor<DTYPE> *ans  = label->GetResult();

    // std::cout << pred << '\n';

    float accuracy = 0.f;

    int pred_index = 0;
    int ans_index  = 0;

    for (int ba = 0; ba < batch; ba++) {
        pred_index = GetMaxIndex(pred, ba, 10);
        ans_index  = GetMaxIndex(ans, ba, 10);

        if (pred_index == ans_index) {
            accuracy += 1.f;
        }
    }

    return (float)(accuracy / batch);
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::GetMaxIndex(Tensor<DTYPE> *data, int ba, int numOfClass) {
    int   index = 0;
    DTYPE max   = (*data)[ba * numOfClass];
    int   start = ba * numOfClass;
    int   end   = ba * numOfClass + numOfClass;

    for (int dim = start + 1; dim < end; dim++) {
        if ((*data)[dim] > max) {
            max   = (*data)[dim];
            index = dim - start;
        }
    }

    return index;
}

template<typename DTYPE> float NeuralNetwork<DTYPE>::GetLoss() {
    float avg_loss = 0.f;

    int batch = m_aLossFunction->GetResult()->GetBatchSize();

    for (int k = 0; k < batch; k++) {
        avg_loss += (*m_aLossFunction)[k] / batch;
    }

    return avg_loss;
}

// ===========================================================================================

template<typename DTYPE> int NeuralNetwork<DTYPE>::ForwardPropagate() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->ForwardPropagate();
    }
    m_aLossFunction->ForwardPropagate();

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::BackPropagate() {
    m_aLossFunction->BackPropagate();

    for (int i = m_OperatorDegree - 1; i >= 0; i--) {
        (*m_aaOperator)[i]->BackPropagate();
    }
    return TRUE;
}

template<typename DTYPE> void *NeuralNetwork<DTYPE>::ForwardPropagate_T(void *param) {
    ThreadInfo *pThreadInfo = (ThreadInfo *)param;

    NeuralNetwork<DTYPE> *pNN = (NeuralNetwork<DTYPE> *)(pThreadInfo->m_NN);
    int pTime                 = pThreadInfo->m_time;
    int pThreadNum            = pThreadInfo->m_threadNum;

    Container<Operator<DTYPE> *> *m_aaOperator = pNN->GetOperatorContainer();
    int m_OperatorDegree                       = m_aaOperator->GetSize();
    LossFunction<DTYPE> *m_aLossFunction             = pNN->GetLossFunction();

    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->ForwardPropagate(pTime, pThreadNum);
    }
    m_aLossFunction->ForwardPropagate(pTime, pThreadNum);
    return NULL;
}

template<typename DTYPE> void *NeuralNetwork<DTYPE>::BackPropagate_T(void *param) {
    ThreadInfo *pThreadInfo = (ThreadInfo *)param;

    NeuralNetwork<DTYPE> *pNN = (NeuralNetwork<DTYPE> *)(pThreadInfo->m_NN);
    int pTime                 = pThreadInfo->m_time;
    int pThreadNum            = pThreadInfo->m_threadNum;

    Container<Operator<DTYPE> *> *m_aaOperator = pNN->GetOperatorContainer();
    int m_OperatorDegree                       = m_aaOperator->GetSize();
    LossFunction<DTYPE> *m_aLossFunction             = pNN->GetLossFunction();

    m_aLossFunction->BackPropagate(pTime, pThreadNum);

    for (int i = m_OperatorDegree - 1; i >= 0; i--) {
        (*m_aaOperator)[i]->BackPropagate(pTime, pThreadNum);
    }
    return NULL;
}

// =========

template<typename DTYPE> int NeuralNetwork<DTYPE>::Training() {
    this->ResetOperatorResult();
    this->ResetOperatorGradient();
    this->ResetLossFunctionResult();
    this->ResetLossFunctionGradient();

    if (m_numOfThread > 1) {
        this->_Training_MT();
    } else if (m_numOfThread == 1) {
        this->ForwardPropagate();
        this->BackPropagate();
    } else return FALSE;

    m_aOptimizer->UpdateVariable();

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::Testing() {
    this->ResetOperatorResult();
    this->ResetLossFunctionResult();

    if (m_numOfThread > 1) {
        this->_Testing_MT();
    } else if (m_numOfThread == 1) {
        this->ForwardPropagate();
    } else return FALSE;

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::_Training_MT() {
    pthread_t  *pThread     = (pthread_t *)malloc(sizeof(pthread_t) * m_numOfThread);
    ThreadInfo *pThreadInfo = (ThreadInfo *)malloc(sizeof(ThreadInfo) * m_numOfThread);

    for (int i = 0; i < m_numOfThread; i++) {
        pThreadInfo[i].m_NN        = (void *)this;
        pThreadInfo[i].m_time      = 0;
        pThreadInfo[i].m_threadNum = i;
        pthread_create(&(pThread[i]), NULL, ForwardPropagate_T, (void *)&(pThreadInfo[i]));
    }

    for (int i = 0; i < m_numOfThread; i++) {
        pthread_join(pThread[i], NULL);
    }

    for (int i = 0; i < m_numOfThread; i++) {
        pthread_create(&(pThread[i]), NULL, BackPropagate_T, (void *)&(pThreadInfo[i]));
    }

    for (int i = 0; i < m_numOfThread; i++) {
        pthread_join(pThread[i], NULL);
    }

    free(pThread);
    free(pThreadInfo);

    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::_Testing_MT() {
    pthread_t  *pThread     = (pthread_t *)malloc(sizeof(pthread_t) * m_numOfThread);
    ThreadInfo *pThreadInfo = (ThreadInfo *)malloc(sizeof(ThreadInfo) * m_numOfThread);

    for (int i = 0; i < m_numOfThread; i++) {
        pThreadInfo[i].m_NN        = (void *)this;
        pThreadInfo[i].m_time      = 0;
        pThreadInfo[i].m_threadNum = i;
        pthread_create(&(pThread[i]), NULL, ForwardPropagate_T, (void *)&(pThreadInfo[i]));
    }

    for (int i = 0; i < m_numOfThread; i++) {
        pthread_join(pThread[i], NULL);
    }

    free(pThread);
    free(pThreadInfo);

    return TRUE;
}

// =========

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeTraining() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->SetModeTraining();
    }
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeAccumulating() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->SetModeAccumulating();
    }
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetModeInferencing() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->SetModeInferencing();
    }
}

#if __CUDNN__

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceGPU() {
    // std::cout << "NeuralNetwork<DTYPE>::SetModeGPU()" << '\n';
    m_Device = GPU;

    for (int i = 0; i < m_OperatorDegree; i++) {
        // important order
        (*m_aaOperator)[i]->SetDeviceGPU();
        (*m_aaOperator)[i]->SetCudnnHandle(m_cudnnHandle);
    }
    m_aLossFunction->SetDeviceGPU();
}

#endif  // __CUDNN__

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceCPU() {
    m_Device = CPU;

    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->SetDeviceCPU();
    }
    m_aLossFunction->SetDeviceCPU();
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::SetDeviceCPU(int pNumOfThread) {
    m_Device      = CPU;
    m_numOfThread = pNumOfThread;

    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->SetDeviceCPU(pNumOfThread);
    }
    m_aLossFunction->SetDeviceCPU(pNumOfThread);
}

// =========

template<typename DTYPE> int NeuralNetwork<DTYPE>::CreateGraph() {
    // in this part, we can check dependency between operator

    return TRUE;
}

template<typename DTYPE> void NeuralNetwork<DTYPE>::PrintGraphInformation() {
    std::cout << "Graph Structure: " << "\n\n";

    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->PrintInformation();
        std::cout << '\n';
    }
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetOperatorResult() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->ResetResult();
    }
    return TRUE;
}

template<typename DTYPE> int NeuralNetwork<DTYPE>::ResetOperatorGradient() {
    for (int i = 0; i < m_OperatorDegree; i++) {
        (*m_aaOperator)[i]->ResetGradient();
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

    for (int i = 0; i < m_OperatorDegree; i++) {
        name = (*m_aaOperator)[i]->GetName();

        if (name == pName) return (*m_aaOperator)[i];
    }

    return NULL;
}
