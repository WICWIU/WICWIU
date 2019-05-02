#ifndef _DATASET_H_
#define _DATASET_H_    value

#include <vector>
#include <queue>

#include "Tensor.hpp"

enum WDATA_TYPE {
    UNKNOWN,
    IMAGE,
    ONE_HOT
};

template<typename DTYPE> class WData {
public:
    DTYPE *m_aData;
    int m_capacity;
    WDATA_TYPE m_type;

    WData(DTYPE *data, int capacity, WDATA_TYPE type = UNKNOWN) {
        m_aData    = data;
        m_capacity = capacity;
        m_type     = type;
    }

    virtual ~WData() {
        delete[] m_aData;
    }

    virtual DTYPE* GetData() {
        return m_aData;
    }

    virtual int GetCapacity() {
        return m_capacity;
    }

    WDATA_TYPE GetType() {
        return m_type;
    }
};

template<typename DTYPE> class Dataset {  // [] operator override
private:
    /* data */
    char m_dataPath[256];
    std::vector<char *> imageName;
    std::vector<int> label;

public:
    Dataset();
    virtual ~Dataset();

    virtual void                         Alloc();
    virtual void                         Dealloc();

    virtual std::vector<WData<DTYPE> *>* GetData(int idx);
    virtual int                          GetLength();
};

template<typename DTYPE> Dataset<DTYPE>::Dataset() {
#ifdef __DEBUG__
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG__
}

template<typename DTYPE> Dataset<DTYPE>::~Dataset() {
#ifdef __DEBUG__
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG__
}

template<typename DTYPE> void Dataset<DTYPE>::Alloc() {
#ifdef __DEBUG__
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG__
}

template<typename DTYPE> void Dataset<DTYPE>::Dealloc() {
#ifdef __DEBUG__
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG__
}

template<typename DTYPE> std::vector<WData<DTYPE> *> *Dataset<DTYPE>::GetData(int idx) {
    // virtual
    // we need to implement default function
}

template<typename DTYPE> int Dataset<DTYPE>::GetLength() {
    // virtual
    // we need to implement default function
    return 1;
}

#endif  // ifndef _DATASET_H_
