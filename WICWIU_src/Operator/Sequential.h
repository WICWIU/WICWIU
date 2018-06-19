#ifndef SEQUENTIAL_H_
#define SEQUENTIAL_H_    value

#include "../Operator.h"

template<typename DTYPE>
class Sequential : public Operator<DTYPE>{
    Operator<DTYPE> **m_listOfOperator;
    int m_numOfOperator;

public:
    Sequential(int numOfOperator, ...) {
        std::cout << "Sequential::Sequential(Operator<DTYPE> *)" << '\n';

        m_listOfOperator = NULL;
        m_numOfOperator  = 0;

        va_list ap;
        va_start(ap, numOfOperator);

        Alloc(numOfOperator, &ap);

        va_end(ap);
    }

    ~Sequential() {
        std::cout << "Sequential::~Sequential()" << '\n';
    }

    int Alloc(int numOfOperator, va_list *ap) {
        std::cout << "Sequential::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        m_listOfOperator = new Operator<DTYPE> *[numOfOperator];
        m_numOfOperator  = numOfOperator;

        for (int i = 0; i < numOfOperator; i++) {
            m_listOfOperator[i] = va_arg(*ap, Operator<DTYPE> *);
        }

        return TRUE;
    }

    int ForwardPropagate() {
        return TRUE;
    }

    int BackPropagate() {
        return TRUE;
    }
};

#endif  // SEQUENTIAL_H_
