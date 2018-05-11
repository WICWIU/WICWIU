#ifndef TENSORHOLDER_H_
#define TENSORHOLDER_H_    value

#include "..//Operator.h"

template<typename DTYPE> class Tensorholder : public Operator<DTYPE>{
private:
    int m_isTrainable;

public:
    Tensorholder(Tensor<DTYPE> *pTensor, std::string pName, int pTrainable = 1) : Operator<DTYPE>(pName) {
        std::cout << "Tensorholder<DTYPE>::Tensorholder(Tensor<DTYPE> *, std::string)" << '\n';
        this->Alloc(pTensor, pTrainable);
    }

    Tensorholder(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, std::string pName) : Operator<DTYPE>(pName) {
        std::cout << "Placeholder<DTYPE>::Placeholder(int, int, int, int, int, std::string)" << '\n';

        this->Alloc(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);
    }

    ~Tensorholder() {
        std::cout << "Tensorholder<DTYPE>::~Tensorholder()" << '\n';
    }

    int Alloc(Tensor<DTYPE> *pTensor, int pTrainable) {
        std::cout << "Tensorholder<DTYPE>::Alloc(Tensor<DTYPE> *, std::string)" << '\n';

        if (pTensor) {
            this->SetResult(pTensor);
        } else {
            printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        this->AddGradient(new Tensor<DTYPE>(new Shape(pTensor->GetShape())));

        return TRUE;
    }

    int Alloc(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
        std::cout << "Placeholder<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';

        Tensor<DTYPE> *pTensor = Tensor<float>::Zeros(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);

        if (pTensor) {
            this->SetResult(pTensor);
        } else {
            printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        Shape *shapeOfDelta = new Shape(pTensor->GetShape());
        this->AddGradient(new Tensor<DTYPE>(shapeOfDelta));

        return TRUE;
    }

    int ForwardPropagate() {
        return TRUE;
    }

    int BackPropagate() {
        return TRUE;
    }

    void SetTensor(Tensor<DTYPE> *pTensor) {
        this->SetResult(pTensor);
    }

};

#endif  // TENSORHOLDER_H_
