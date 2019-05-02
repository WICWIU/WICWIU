#ifndef _DATASET_H_
#define _DATASET_H_    value

#include <vector>
#include <queue>

#include "Tensor.hpp"

template<typename DTYPE> class WData {
public:
    WData();
    virtual ~WData();
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
    return 0;
}

#endif  // ifndef _DATASET_H_
