#include "Common.h"

template<typename DTYPE> class Tensor;
template<typename DTYPE> class Operator;
template<typename DTYPE> class Tensorholder;

/*!
@class Container Operator와 Tensor를 저장하기 위한 Queue에 해당하는 클래스
@details Tensor, Operator, Tensorholder 세 가지 클래스에 대한 Queue를 동적으로 할당한다.
@details 기본 queue 구조에 인덱스를 이용한 접근 및 역순으로 접근 등 추가적인 메소드가 구현되어 있다.
*/
template<typename DTYPE> class Container {
private:
    DTYPE *m_aElement;
    ///< 동적으로 할당받는 Queue의 Element 배열
    int m_size;
    ///< 동적으로 할당받는 Queue의 Element 개수

public:
    /*!
    @brief Container 생성자
    @details 각 멤버 변수를 초기화하여 Container 클래스를 생성한다.
    @return 없음
    */
    Container() {
        #ifdef __DEBUG__
        std::cout << "Container<DTYPE>::Container()" << '\n';
        #endif  // __DEBUG__
        m_aElement = NULL;
        m_size     = 0;
    }

    /*!
    @brief Container 클래스 소멸자
    @details 해당 Container 클래스를 위해 동적으로 할당된 메모리 공간을 반환하고 클래스를 소멸한다.
    @return 없음
    */
    virtual ~Container() {
        if (m_aElement) {
            delete[] m_aElement;
            m_aElement = NULL;
        }
    }

    /*!
    @brief Queue의 push 메소드
    @details 기존의 queue를 할당 해제하고 매개변수로 받은 Element를 마지막에 추가하여 새로운 Queue를 동적으로 할당한다.
    @param pElement Queue에 추가하고자 하는 변수
    @return 성공 시 TRUE, 실패 시 FALSE
    */
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

    /*!
    @brief Queue의 pop 메소드
    @details 기존의 queue를 할당 해제하고 Queue의 첫번째 Element를 반환한 후 새로운 Queue를 동적으로 할당한다.
    @return Queue의 첫번째 Element
    */
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

    /*!
    @brief Queue에서 Element를 찾아 반환하는 pop 메소드
    @details 매개변수로 받은 Element가 Queue에 존재할 경우, 해당 Element를 반환하고 Queue를 새로 동적으로 할당한다.
    @param pElement Queue에서 찾고자 하는 Element
    @return 실패 시 NULL, 성공 시 매개변수로 전달받은 Element와 동일한 Queue의 Element
    */
    DTYPE Pop(DTYPE pElement) {
        int index = -1;

        for (int i = 0; i < m_size; i++) {
            if (m_aElement[i] == pElement) index = i;
        }

        if (index == -1) {
            std::cout << "There is no element!" << '\n';
            return NULL;
        }

        // DTYPE  element = m_aElement[index];
        DTYPE *temp = new DTYPE[m_size - 1];

        for (int i = 0, j = 0; i < m_size - 1;) {
            if (index != j) {
                temp[i] = m_aElement[j];
                i++;
            }

            j++;
        }

        if (m_aElement) {
            delete[] m_aElement;
            m_aElement = NULL;
        }

        m_aElement = temp;

        m_size--;

        return pElement;
    }

    /*!
    @brief Queue를 역순으로 재할당해주는 메소드
    @details Queue의 Element를 반대 순서로 저장하는 새로운 Queue를 할당하고, 기존의 Queue를 할당 해제한다.
    @return TRUE
    */
    int Reverse() {
        DTYPE *temp = new DTYPE[m_size];

        for (int i = 0; i < m_size; i++) temp[m_size - i - 1] = m_aElement[i];

        if (m_aElement) {
            delete[] m_aElement;
            m_aElement = NULL;
        }

        m_aElement = temp;

        return TRUE;
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

    /*!
    @brief []연산자 오버로딩
    @details Queue에서 파라미터로 받은 인덱스에 해당하는 ELement를 반환한다.
    @param index 찾고자 하는 Queue의 Element의 인덱스
    @return m_aElement[index]
    */
    DTYPE& operator[](unsigned int index) {
        return m_aElement[index];
    }
};
