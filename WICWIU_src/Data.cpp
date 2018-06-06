#include "Data.h"

template class Data<int>;
template class Data<float>;
template class Data<double>;

template<typename DTYPE> Data<DTYPE>::Data() {
    m_timeSize        = 0;
    m_capacityPerTime = 0;
    m_aHostData       = NULL;
#if __CUDNN__
    m_aDevData = NULL;
    m_Device   = CPU;
#endif  // __CUDNN
}

template<typename DTYPE> Data<DTYPE>::Data(unsigned int pTimeSize, unsigned int pCapacity) {
    #if __DEBUG__
    std::cout << "Data<DTYPE>::Data(Shape *)" << '\n';
    #endif  // __DEBUG__
    m_timeSize        = pTimeSize;
    m_capacityPerTime = pCapacity;
    m_aHostData       = NULL;
#if __CUDNN__
    m_aDevData = NULL;
#endif  // __CUDNN
    m_Device = CPU;
    Alloc();
}

template<typename DTYPE> Data<DTYPE>::Data(Data *pData) {
    #if __DEBUG__
    std::cout << "Data<DTYPE>::Data(Data *)" << '\n';
    #endif  // __DEBUG__
    m_timeSize        = pData->GetTimeSize();
    m_capacityPerTime = pData->GetCapacityPerTime();
    m_aHostData       = NULL;
#if __CUDNN__
    m_aDevData = NULL;
#endif  // __CUDNN
    m_Device = CPU;
    Alloc(pData);
}

template<typename DTYPE> Data<DTYPE>::~Data() {
    #if __DEBUG__
    std::cout << "Data<DTYPE>::~Data()" << '\n';
    #endif  // __DEBUG__
    Delete();
}

template<typename DTYPE> int Data<DTYPE>::Alloc() {
    m_aHostData = new DTYPE *[m_timeSize];

    for (int i = 0; i < m_timeSize; i++) {
        m_aHostData[i] = new DTYPE[m_capacityPerTime];

        for (int j = 0; j < m_capacityPerTime; j++) {
            m_aHostData[i][j] = 0.f;
        }
    }

    return TRUE;
}

template<typename DTYPE> int Data<DTYPE>::Alloc(Data *pData) {
    m_aHostData = new DTYPE *[m_timeSize];

    for (int i = 0; i < m_timeSize; i++) {
        m_aHostData[i] = new DTYPE[m_capacityPerTime];

        for (int j = 0; j < m_capacityPerTime; j++) {
            m_aHostData[i][j] = (*pData)[i * m_capacityPerTime + j];
        }
    }

    return TRUE;
}

template<typename DTYPE> void Data<DTYPE>::Delete() {
    if (m_aHostData) {
        for (int i = 0; i < m_timeSize; i++) {
            if (m_aHostData[i]) {
                delete[] m_aHostData[i];
                m_aHostData[i] = NULL;
            }
        }
        delete[] m_aHostData;
        m_aHostData = NULL;
    }
}

template<typename DTYPE> int Data<DTYPE>::GetCapacity() {
    return m_timeSize * m_capacityPerTime;
}

template<typename DTYPE> int Data<DTYPE>::GetTimeSize() {
    return m_timeSize;
}

template<typename DTYPE> int Data<DTYPE>::GetCapacityPerTime() {
    return m_capacityPerTime;
}

template<typename DTYPE> DTYPE& Data<DTYPE>::operator[](unsigned int index) {
    return m_aHostData[index / m_capacityPerTime][index % m_capacityPerTime];
}

template<typename DTYPE> DTYPE *Data<DTYPE>::GetHostData(unsigned int pTime) {
    return m_aHostData[pTime];
}

#ifdef __CUDNN__

template<typename DTYPE> DTYPE *Data<DTYPE>::GetDeviceData(unsigned int pTime) {
    return m_aDevData[pTime];
}

template<typename DTYPE> void Data<DTYPE>::MemcpyDeviceToHost() {
    if (m_aDevData != NULL) {
        for (int i = 0; i < m_timeSize; i++) {
            checkCudaErrors(cudaMemcpy(m_aHostData[i], m_aDevData[i], (m_capacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToHost));
        }
    }
}

template<typename DTYPE> void Data<DTYPE>::MemcpyHostToDevice() {
    if (m_aDevData == NULL) {
        m_aDevData = new DTYPE *[m_timeSize];

        for (int i = 0; i < m_timeSize; i++) {
            checkCudaErrors(cudaMalloc((void **)&(m_aDevData[i]), (m_capacityPerTime * sizeof(DTYPE))));
            checkCudaErrors(cudaMemcpy(m_aDevData[i], m_aHostData[i], (m_capacityPerTime * sizeof(DTYPE)), cudaMemcpyHostToDevice));
        }
    } else {
        for (int i = 0; i < m_timeSize; i++) {
            checkCudaErrors(cudaMemcpy(m_aDevData[i], m_aHostData[i], (m_capacityPerTime * sizeof(DTYPE)), cudaMemcpyHostToDevice));
        }
    }
}

#endif  // if __CUDNN__

//// example code
// int main(int argc, char const *argv[]) {
// Data<int> *pData = new Data<int>(2048);
//
// std::cout << pData->GetCapacity() << '\n';
// std::cout << (*pData)[2048] << '\n';
//
// delete pData;
//
// return 0;
// }
