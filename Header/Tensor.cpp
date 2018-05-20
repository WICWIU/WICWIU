#include "Tensor.h"

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;

////////////////////////////////////////////////////////////////////Class Tensor
template<typename DTYPE> Tensor<DTYPE>::Tensor() {
    #if __DEBUG__
    std::cout << "Tensor::Tensor()" << '\n';
    #endif  // __DEBUG__

    m_aShape = NULL;
    m_aData  = NULL;
    m_Device = CPU;
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
    #if __DEBUG__
    std::cout << "Tensor::Tensor(int, int, int, int, int)" << '\n';
    #endif  // __DEBUG__
    m_aShape = NULL;
    m_aData  = NULL;
    m_Device = CPU;
    Alloc(new Shape(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize));
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(Shape *pShape) {
    #if __DEBUG__
    std::cout << "Tensor::Tensor(Shape*)" << '\n';
    #endif  // __DEBUG__
    m_aShape = NULL;
    m_aData  = NULL;
    m_Device = CPU;
    Alloc(pShape);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(Tensor *pTensor) {
    #if __DEBUG__
    std::cout << "Tensor::Tensor(Shape*)" << '\n';
    #endif  // __DEBUG__
    m_aShape = NULL;
    m_aData  = NULL;
    m_Device = CPU;
    Alloc(pTensor);
}

template<typename DTYPE> Tensor<DTYPE>::~Tensor() {
    #if __DEBUG__
    std::cout << "Tensor::~Tensor()" << '\n';
    #endif  // __DEBUG__
    Delete();
}

template<typename DTYPE> int Tensor<DTYPE>::Alloc(Shape *pShape) {
    if (pShape == NULL) {
        printf("Receive NULL pointer of Shape class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape = pShape;

        int rank = pShape->GetRank();

        if (rank < 5) {
            delete m_aShape;
            m_aShape = NULL;
            printf("Receive invalid rank value %d in %s (%s %d)\n", rank, __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        } else {
            int pTime            = (*pShape)[0];
            int pCapacityPerTime = 1;

            for (int i = 0; i < rank; i++) {
                pCapacityPerTime *= (*pShape)[i];
            }
            m_aData = new Data<DTYPE>(pTime, pCapacityPerTime);
        }
    }

    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::Alloc(Tensor<DTYPE> *pTensor) {
    if (pTensor == NULL) {
        printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape = new Shape(pTensor->GetShape());
        m_aData  = new Data<DTYPE>(pTensor->GetData());
    }

    return TRUE;
}

template<typename DTYPE> void Tensor<DTYPE>::Delete() {
    if (m_aShape) {
        delete m_aShape;
        m_aShape = NULL;
    }

    if (m_aData) {
        delete m_aData;
        m_aData = NULL;
    }
}

template<typename DTYPE> Shape *Tensor<DTYPE>::GetShape() {
    return m_aShape;
}

template<typename DTYPE> Data<DTYPE> *Tensor<DTYPE>::GetData() {
    return m_aData;
}

template<typename DTYPE> int Tensor<DTYPE>::GetTimeSize() {
    return (*m_aShape)[0];
}

template<typename DTYPE> int Tensor<DTYPE>::GetBatchSize() {
    return (*m_aShape)[1];
}

template<typename DTYPE> int Tensor<DTYPE>::GetChannelSize() {
    return (*m_aShape)[2];
}

template<typename DTYPE> int Tensor<DTYPE>::GetRowSize() {
    return (*m_aShape)[3];
}

template<typename DTYPE> int Tensor<DTYPE>::GetColSize() {
    return (*m_aShape)[4];
}

template<typename DTYPE> int Tensor<DTYPE>::GetCapacity() {
    return m_aData->GetCapacity();
}

template<typename DTYPE> DTYPE *Tensor<DTYPE>::GetHostData(unsigned int pTime) {
    #if __CUDNN__
    # if __DEBUG__

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->MemcpyDeviceToHost();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->MemcpyDeviceToHost();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    return m_aData->GetHostData(pTime);
}

#ifdef __CUDNN__

template<typename DTYPE> DTYPE *Tensor<DTYPE>::GetDeviceData(unsigned int pTime) {
    # if __DEBUG__

    if (m_Device == CPU) {
        printf("Warning! Tensor is allocated in Host(CPU) latest time\n");
        printf("Change mode CPU toGPU\n");
        this->MemcpyHostToDevice();
    }

    # else // if __DEBUG__

    if (m_Device == CPU) {
        this->MemcpyHostToDevice();
    }

    # endif // __DEBUG__

    return m_aData->GetDeviceData(pTime);
}

template<typename DTYPE> void Tensor<DTYPE>::MemcpyDeviceToHost() {
    m_Device = CPU;
    m_aData->MemcpyDeviceToHost();
}

template<typename DTYPE> void Tensor<DTYPE>::MemcpyHostToDevice() {
    m_Device = GPU;
    m_aData->MemcpyHostToDevice();
}

template<typename DTYPE> cudnnTensorDescriptor_t& Tensor<DTYPE>::GetDescriptor() {
    return m_aShape->GetDescriptor();
}

#endif  // if __CUDNN__


//////////////////////////////////////////////////////////////////

template<typename DTYPE> int Tensor<DTYPE>::Reshape(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
    int cur_capacity = GetCapacity();
    int new_capacity = pTimeSize * pBatchSize * pChannelSize * pRowSize * pColSize;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot Reshape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape->ReShape(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);
    }

    return TRUE;
}

template<typename DTYPE> void Tensor<DTYPE>::Reset() {
    int capacity = GetCapacity();

    #if __CUDNN__
    # if __DEBUG__

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->MemcpyDeviceToHost();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->MemcpyDeviceToHost();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    for (int i = 0; i < capacity; i++) {
        (*m_aData)[i] = 0;
    }
}

#if __CUDNN__

template<typename DTYPE> void Tensor<DTYPE>::Reset(cudnnHandle_t& pCudnnHandle) {
    int pTime = this->GetTimeSize();
    cudnnTensorDescriptor_t pDesc = this->GetDescriptor();
    DTYPE * pDevData = NULL;
    float zero = 0.f;

    for(int i = 0; i < pTime; i++){
        pDevData = this->GetDeviceData(i);
        checkCUDNN(cudnnAddTensor(pCudnnHandle,
                                  &zero, pDesc, pDevData,
                                  &zero, pDesc, pDevData));
    }

}

#endif // if __CUDNN__


///////////////////////////////////////////////////////////////////

template<typename DTYPE> DTYPE& Tensor<DTYPE>::operator[](unsigned int index) {
    #if __CUDNN__
    # if __DEBUG__

    if (m_Device == GPU) {
        printf("Warning! Tensor is allocated in Device(GPU) latest time\n");
        printf("Change mode GPU to CPU\n");
        this->MemcpyDeviceToHost();
    }

    # else // if __DEBUG__

    if (m_Device == GPU) {
        this->MemcpyDeviceToHost();
    }

    # endif // __DEBUG__
    #endif  // __CUDNN__

    return (*m_aData)[index];
}

//////////////////////////////////////////////////////////////////static method

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Truncated_normal(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, float mean, float stddev) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Truncated_normal()" << '\n';
    #endif  // __DEBUG__

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> rand(mean, stddev);

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);

    int capacity = temp->GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*temp)[i] = rand(gen);
    }

    return temp;
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Zeros(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Zero()" << '\n';
    #endif  // __DEBUG__

    return new Tensor<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Constants(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, DTYPE constant) {
    #if __DEBUG__
    std::cout << "Tensor<DTYPE>::Constant()" << '\n';
    #endif  // __DEBUG__

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);

    int capacity = temp->GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*temp)[i] = constant;
    }

    return temp;
}

// example code
// int main(int argc, char const *argv[]) {
// Tensor<float> *left  = Tensor<float>::Constants(1, 2, 3, 3, 3, 2);
// Tensor<float> *right = Tensor<float>::Truncated_normal(1, 1, 3, 1, 1, 0.0, 0.1);
// Tensor<float> *dst   = Tensor<float>::Zeros(1, 2, 3, 3, 3);
//
// std::cout << left << '\n';
// std::cout << right << '\n';
// std::cout << dst << '\n';
//
// Tensor<float>::BroadcastAdd(left, right);
//
// std::cout << dst << '\n';
//
// return 0;
// }
