#include "Common.h"

template<typename DTYPE> class Tensor;
template<typename DTYPE> class Operator;
template<typename DTYPE> class Tensorholder;

template<typename DTYPE> class Container {
private:
    DTYPE *m_aElement;
    int m_size;

public:
    Container() {
        #if __DEBUG__
        std::cout << "Container<DTYPE>::Container()" << '\n';
        #endif  // __DEBUG__
        m_aElement = NULL;
        m_size     = 0;
    }

    virtual ~Container() {
        if (m_aElement) {
            delete[] m_aElement;
            m_aElement = NULL;
        }
    }

    int Push(DTYPE pElement) {
        try {
            DTYPE *temp = new DTYPE[m_size + 1];

            for (int i = 0; i < m_size; i++) temp[i] = m_aElement[i];
            temp[m_size] = pElement;

            if (m_aElement) {
                delete[] m_aElement;
                m_aElement = NULL;
            }

            m_aElement = temp;
        } catch (...) {
            printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        m_size++;

        return TRUE;
    }

    DTYPE Pop() {
        DTYPE  element = m_aElement[0];
        DTYPE *temp    = new DTYPE[m_size - 1];

        for (int i = 1; i < m_size; i++) temp[i - 1] = m_aElement[i];

        if (m_aElement) {
            delete[] m_aElement;
            m_aElement = NULL;
        }

        m_aElement = temp;

        m_size--;

        return element;
    }

    int SetElement(DTYPE pElement, unsigned int index) {
        m_aElement[index] = pElement;
        return TRUE;
    }

    int GetSize() {
        // std::cout << "Container<DTYPE>::GetSize()" << '\n';
        return m_size;
    }

    DTYPE GetLast() {
        // std::cout << "Container<DTYPE>::GetLast()" << '\n';
        return m_aElement[m_size - 1];
    }

    DTYPE* GetRawData() const {
        return m_aElement;
    }

    DTYPE GetElement(unsigned int index) {
        return m_aElement[index];
    }

    DTYPE operator[](unsigned int index) {
        return m_aElement[index];
    }
};
