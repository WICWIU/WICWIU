#ifndef _DATASET_H_
#define _DATASET_H_    value

#include <vector>
#include <queue>

#include "Tensor.hpp"

template<typename DTYPE> class WData {
public:
    DTYPE *m_aData;
    int m_capacity;

    WData(DTYPE *data, int capacity) {
        m_aData    = data;
        m_capacity = capacity;
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

    DTYPE& operator[](int idx) {
        return m_aData[idx];
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
    int                                  GetNumOfDatasetMember();
};

template<typename DTYPE> Dataset<DTYPE>::Dataset() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
}

template<typename DTYPE> Dataset<DTYPE>::~Dataset() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
}

template<typename DTYPE> void Dataset<DTYPE>::Alloc() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
}

template<typename DTYPE> void Dataset<DTYPE>::Dealloc() {
#ifdef __DEBUG___
    std::cout << __FUNCTION__ << '\n';
    std::cout << __FILE__ << '\n';
#endif  // ifdef __DEBUG___
}

template<typename DTYPE> std::vector<WData<DTYPE> *> *Dataset<DTYPE>::GetData(int idx) {
    // virtual
    // we need to implement default function
    std::vector<WData<DTYPE> *> *result = new std::vector<WData<DTYPE> *>(1, NULL);
    int capacity                        = 1;

    DTYPE *_data = new DTYPE[capacity];
    _data[0] = idx;

    WData<DTYPE> *data = new WData<DTYPE>(_data, capacity);
    (*result)[0] = data;

    return result;
}

template<typename DTYPE> int Dataset<DTYPE>::GetLength() {
    // virtual
    // we need to implement default function
    return 100;
}

template<typename DTYPE> int Dataset<DTYPE>::GetNumOfDatasetMember() {
    // virtual
    // we need to implement default function
    int numOfDatasetMember = 0;

    std::vector<WData<DTYPE> *> *temp = this->GetData(0);

    if (temp) {
        numOfDatasetMember = temp->size();

        for (int i = 0; i < numOfDatasetMember; i++) {
            if ((*temp)[i]) {
                delete (*temp)[i];
                (*temp)[i] = NULL;
            }
        }
        delete temp;
        temp = NULL;
    }


    return numOfDatasetMember;
}

#endif  // ifndef _DATASET_H_
