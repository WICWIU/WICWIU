#ifndef _DATASET_H_
#define _DATASET_H_    value

#include <vector>
#include <queue>

#include "Tensor.hpp"

template<typename DTYPE> class Dataset {  // [] operator override
private:
    /* data */
    char m_dataPath[256];
    std::vector<char *> imageName;
    std::vector<int> label;

public:
    Dataset();
    virtual ~Dataset();

    virtual void                          Alloc();
    virtual void                          Dealloc();

    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);
    virtual int                           GetLength();
    int                                   GetNumOfDatasetMember();
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

template<typename DTYPE> std::vector<Tensor<DTYPE> *> *Dataset<DTYPE>::GetData(int idx) {
    // virtual
    // we need to implement default function
    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(1, NULL);
    int capacity                         = 1;

    Tensor<DTYPE> *data = Tensor<DTYPE>::Zeros(1, 1, 1, 1, capacity);
    (*data)[0]   = (DTYPE)idx;
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

    std::vector<Tensor<DTYPE> *> *temp = this->GetData(0);

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
