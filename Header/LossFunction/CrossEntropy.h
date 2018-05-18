#ifndef CROSSENTROPY_H_
#define CROSSENTROPY_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class CrossEntropy : public Operator<DTYPE>{
private:
    DTYPE m_epsilon = 0.0;  // for backprop

public:
    CrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, int epsilon = 1e-30) : Operator<DTYPE>(pInput0, pInput1) {
        std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        this->Alloc(pInput0, pInput1, epsilon);
    }

    CrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        this->Alloc(pInput0, pInput1);
    }

    CrossEntropy(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, int epsilon, std::string pName) : Operator<DTYPE>(pInput0, pInput1, pName) {
        std::cout << "CrossEntropy::CrossEntropy(Operator<DTYPE> *, Operator<DTYPE> *, int, std::string)" << '\n';
        this->Alloc(pInput0, pInput1, epsilon);
    }

    ~CrossEntropy() {
        std::cout << "CrossEntropy::~CrossEntropy()" << '\n';
    }

    virtual int Alloc(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, int epsilon = 1e-30) {
        std::cout << "CrossEntropy::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';

        return 1;
    }

    virtual int ForwardPropagate() {
        return 1;
    }

    virtual int BackPropagate() {
        return 1;
    }
};

#endif  // CROSSENTROPY_H_
