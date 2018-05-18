#include "Data.h"

template class Data<int>;
template class Data<float>;
template class Data<double>;

template<typename DTYPE> Data<DTYPE>::Data() {
    m_timeSize        = 0;
    m_capacityPerTime = 0;
    m_aData           = NULL;
}

template<typename DTYPE> Data<DTYPE>::Data(unsigned int pTimeSize, unsigned int pCapacity) {
    // std::cout << "Data<DTYPE>::Data(Shape *)" << '\n';
    m_timeSize        = pTimeSize;
    m_capacityPerTime = pCapacity;
    m_aData           = NULL;
    Alloc();
}

template<typename DTYPE> Data<DTYPE>::Data(Data *pData) {
    std::cout << "Data<DTYPE>::Data(Data *)" << '\n';
    m_timeSize        = pData->GetTimeSize();
    m_capacityPerTime = pData->GetCapacityPerTime();
    m_aData           = NULL;
    Alloc(pData);
}

template<typename DTYPE> Data<DTYPE>::~Data() {
    // std::cout << "Data<DTYPE>::~Data()" << '\n';
    Delete();
}

template<typename DTYPE> int Data<DTYPE>::Alloc() {
    m_aData = new DTYPE *[m_timeSize];

    for (int i = 0; i < m_timeSize; i++) {
        m_aData[i] = new DTYPE[m_capacityPerTime];

        for (int j = 0; j < m_capacityPerTime; j++) {
            m_aData[i][j] = 0.f;
        }
    }

    return TRUE;
}

template<typename DTYPE> int Data<DTYPE>::Alloc(Data *pData) {
    m_aData = new DTYPE *[m_timeSize];

    for (int i = 0; i < m_timeSize; i++) {
        m_aData[i] = new DTYPE[m_capacityPerTime];

        for (int j = 0; j < m_capacityPerTime; j++) {
            m_aData[i][j] = (*pData)[i * m_capacityPerTime + j];
        }
    }

    return TRUE;
}

template<typename DTYPE> void Data<DTYPE>::Delete() {
    if (m_aData) {
        for (int i = 0; i < m_timeSize; i++) {
            if (m_aData[i]) {
                delete[] m_aData[i];
                m_aData[i] = NULL;
            }
        }
        delete[] m_aData;
        m_aData = NULL;
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
    return m_aData[index / m_capacityPerTime][index % m_capacityPerTime];
}

template<typename DTYPE> DTYPE* Data<DTYPE>::GetLowData(unsigned int pTime) {
    return m_aData[pTime];
}

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
