#include "Data.h"

template class Data<int>;
template class Data<float>;
template class Data<double>;

template<typename DTYPE> Data<DTYPE>::Data() {
    m_Capacity = 0;
    m_Cols     = 0;
    m_Rows     = 0;
    m_aData    = NULL;
}

template<typename DTYPE> Data<DTYPE>::Data(unsigned int pCapacity) {
    // std::cout << "Data<DTYPE>::Data(Shape *)" << '\n';
    m_Capacity = 0;
    m_Cols     = 0;
    m_Rows     = 0;
    m_aData    = NULL;
    Alloc(pCapacity);
}

template<typename DTYPE> Data<DTYPE>::Data(Data *pData) {
    std::cout << "Data<DTYPE>::Data(Data *)" << '\n';
    m_Capacity = 0;
    m_Cols     = 0;
    m_Rows     = 0;
    m_aData    = NULL;
    Alloc(pData);
}

template<typename DTYPE> Data<DTYPE>::~Data() {
    // std::cout << "Data<DTYPE>::~Data()" << '\n';
    Delete();
}

template<typename DTYPE> int Data<DTYPE>::Alloc(unsigned int pCapacity) {
    m_Capacity = pCapacity;
    m_Cols     = SIZEOFCOLS;

    if (m_Capacity % SIZEOFCOLS != 0) {
        m_Rows  = m_Capacity / SIZEOFCOLS + 1;
        m_aData = new DTYPE *[m_Rows];

        for (int i = 0; i < m_Rows; i++) {
            if (i != (m_Rows - 1)) {
                m_aData[i] = new DTYPE[SIZEOFCOLS];

                for (int j = 0; j < SIZEOFCOLS; j++) {
                    m_aData[i][j] = 0.f;
                }
            } else {
                int cols = m_Capacity % SIZEOFCOLS;

                m_aData[i] = new DTYPE[cols];

                for (int j = 0; j < cols; j++) {
                    m_aData[i][j] = 0.f;
                }
            }
        }
    } else {
        m_Rows  = m_Capacity / SIZEOFCOLS;
        m_aData = new DTYPE *[m_Rows];

        for (int i = 0; i < m_Rows; i++) {
            m_aData[i] = new DTYPE[SIZEOFCOLS];

            for (int j = 0; j < SIZEOFCOLS; j++) {
                m_aData[i][j] = 0.f;
            }
        }
    }

    return TRUE;
}

template<typename DTYPE> int Data<DTYPE>::Alloc(Data *pData) {
    m_Capacity = pData->GetCapacity();
    m_Cols     = SIZEOFCOLS;

    if (m_Capacity % SIZEOFCOLS != 0) {
        m_Rows  = m_Capacity / SIZEOFCOLS + 1;
        m_aData = new DTYPE *[m_Rows];

        for (int i = 0; i < m_Rows; i++) {
            if (i != (m_Rows - 1)) {
                m_aData[i] = new DTYPE[SIZEOFCOLS];

                for (int j = 0; j < SIZEOFCOLS; j++) {
                    m_aData[i][j] = (*pData)[i * SIZEOFCOLS + j];
                }
            } else {
                int cols = m_Capacity % SIZEOFCOLS;

                m_aData[i] = new DTYPE[cols];

                for (int j = 0; j < cols; j++) {
                    m_aData[i][j] = (*pData)[i * SIZEOFCOLS + j];
                }
            }
        }
    } else {
        m_Rows  = m_Capacity / SIZEOFCOLS;
        m_aData = new DTYPE *[m_Rows];

        for (int i = 0; i < m_Rows; i++) {
            m_aData[i] = new DTYPE[SIZEOFCOLS];

            for (int j = 0; j < SIZEOFCOLS; j++) {
                m_aData[i][j] = (*pData)[i * SIZEOFCOLS + j];
            }
        }
    }

    return TRUE;
}

template<typename DTYPE> void Data<DTYPE>::Delete() {
    if (m_aData) {
        for (int i = 0; i < m_Rows; i++) {
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
    return m_Capacity;
}

template<typename DTYPE> DTYPE& Data<DTYPE>::operator[](unsigned int index) {
    return m_aData[index / SIZEOFCOLS][index % SIZEOFCOLS];
}

template<typename DTYPE> DTYPE& Data<DTYPE>::GetRawData() {
    return **m_aData;
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
