#ifndef SOFTMAX_H_
#define SOFTMAX_H_    value

#include "../Operator.h"

template<typename DTYPE>
class Softmax : public Operator<DTYPE>{
    DTYPE m_epsilon = 0.0;

public:
    // Constructor의 작업 순서는 다음과 같다.
    // 상속을 받는 Operator(Parent class)의 Alloc()을 실행하고, (Operator::Alloc())
    // 나머지 MetaParameter에 대한 Alloc()을 진행한다. (Softmax::Alloc())
    Softmax(Operator<DTYPE> *pInput, DTYPE epsilon = 1e-20) : Operator<DTYPE>(pInput) {
        #ifdef __DEBUG__
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
        #endif  // __DEBUG__
        Alloc(pInput, epsilon);
    }

    Softmax(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
        #endif  // __DEBUG__
        Alloc(pInput);
    }

    Softmax(Operator<DTYPE> *pInput, DTYPE epsilon, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
        #endif  // __DEBUG__
        Alloc(pInput, epsilon);
    }

    ~Softmax() {
        #ifdef __DEBUG__
        std::cout << "Softmax::~Softmax()" << '\n';
        #endif  // __DEBUG__
    }

    virtual int Alloc(Operator<DTYPE> *pInput, DTYPE epsilon = 1e-20) {
        return 1;
    }

    virtual int ForwardPropagate(int pTime = 0) {
        return 1;
    }

    virtual int BackPropagate(int pTime = 0) {
        return 1;
    }
};

#endif  // SOFTMAX_H_
