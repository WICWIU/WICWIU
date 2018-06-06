#ifndef CROSSENTROPY_H_
#define CROSSENTROPY_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class CrossEntropy : public Operator<DTYPE>{
private:
    DTYPE m_epsilon = 0.0;  // for backprop

public:
    CrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, int epsilon = 1e-30) : Operator<DTYPE>(pInput0, pInput1) {
        #if __DEBUG__
        std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput0, pInput1, epsilon);
    }

    CrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        #if __DEBUG__
        std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput0, pInput1);
    }

    CrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, int epsilon, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        #if __DEBUG__
        std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput0, pInput1, epsilon);
    }

    ~CrossEntropy() {
        #if __DEBUG__
        std::cout << "CrossEntropy::~CrossEntropy()" << '\n';
        #endif  // __DEBUG__
    }

    virtual int Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, int epsilon = 1e-30) {
        #if __DEBUG__
        std::cout << "CrossEntropy::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__

        return 1;
    }

    virtual int ForwardPropagate(int pTime = 0, int pThreadNum = 0) {
        return 1;
    }

    virtual int BackPropagate(int pTime = 0, int pThreadNum = 0) {
        return 1;
    }
};

#endif  // CROSSENTROPY_H_
