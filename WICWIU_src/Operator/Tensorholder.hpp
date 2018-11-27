#ifndef TENSORHOLDER_H_
#define TENSORHOLDER_H_    value

#include "../Operator.hpp"

template<typename DTYPE> class Tensorholder : public Operator<DTYPE>{
public:
    Tensorholder(Tensor<DTYPE> *pTensor, std::string pName, int pTrainable = TRUE) : Operator<DTYPE>(pName) {
        #ifdef __DEBUG__
        std::cout << "Tensorholder<DTYPE>::Tensorholder(Tensor<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pTensor, pTrainable);
    }

    Tensorholder(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, std::string pName, int pTrainable = TRUE) : Operator<DTYPE>(pName) {
        #ifdef __DEBUG__
        std::cout << "Placeholder<DTYPE>::Placeholder(int, int, int, int, int, std::string)" << '\n';
        #endif  // __DEBUG__

        this->Alloc(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize, pTrainable);
    }

    ~Tensorholder() {
        #ifdef __DEBUG__
        std::cout << "Tensorholder<DTYPE>::~Tensorholder()" << '\n';
        #endif  // __DEBUG__
    }

    int Alloc(Tensor<DTYPE> *pTensor, int pTrainable) {
        #ifdef __DEBUG__
        std::cout << "Tensorholder<DTYPE>::Alloc(Tensor<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__

        if (pTensor) {
            this->SetResult(pTensor);
        } else {
            printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        this->SetIsTensorholder(TRUE);
        this->SetIsTrainable(pTrainable);
        this->AddGradient(new Tensor<DTYPE>(new Shape(pTensor->GetShape())));

        return TRUE;
    }

    int Alloc(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, int pTrainable) {
        #ifdef __DEBUG__
        std::cout << "Placeholder<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        Tensor<DTYPE> *pTensor = Tensor<float>::Zeros(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);

        if (pTensor) {
            this->SetResult(pTensor);
        } else {
            printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        this->SetIsTensorholder(TRUE);
        this->SetIsTrainable(pTrainable);
        Shape *shapeOfDelta = new Shape(pTensor->GetShape());
        this->AddGradient(new Tensor<DTYPE>(shapeOfDelta));

        return TRUE;
    }

    void SetTensor(Tensor<DTYPE> *pTensor) {
        this->SetResult(pTensor);
    }

    void FeedTensor(Tensor<DTYPE> *pTensor) {
        this->SetResult(pTensor);
    }
};

#endif  // TENSORHOLDER_H_
