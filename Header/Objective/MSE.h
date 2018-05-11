#ifndef MSE_H_
#define MSE_H_    value

#include "..//Objective.h"

template<typename DTYPE>
class MSE : public Objective<DTYPE>{
public:
    MSE(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) : Objective<DTYPE>(pOperator, pLabel, pName) {
        std::cout << "MSE::MSE(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        this->Alloc(pOperator);
    }

    virtual ~MSE() {
        std::cout << "MSE::~MSE()" << '\n';
    }

    virtual int Alloc(Operator<DTYPE> *pOperator) {
        std::cout << "MSE::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        this->SetResult(
            new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1)
            );

        this->SetGradient(
            new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize)
            );

        return TRUE;
    }

    virtual Tensor<DTYPE>* ForwardPropagate() {
        Tensor<DTYPE> *input    = this->GetTensor();
        Tensor<DTYPE> *label    = this->GetLabel()->GetResult();
        Tensor<DTYPE> *result   = this->GetResult();
        Tensor<DTYPE> *gradient = this->GetGradient();
        // result->Reset();
        // gradient->Reset();

        int timesize  = input->GetTimeSize();
        int batchsize = input->GetBatchSize();
        int count     = timesize * batchsize;

        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int index = 0;

        for (int i = 0; i < count; i++) {
            for (int j = 0; j < capacity; j++) {
                index              = i * capacity + j;
                (*result)[i]      += Error((*input)[index], (*label)[index]);
                (*gradient)[index] = ((*input)[index] - (*label)[index]);
            }
        }

        return result;
    }

    virtual Tensor<DTYPE>* BackPropagate() {
        Tensor<DTYPE> *gradient    = this->GetGradient();
        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        int batchsize = gradient->GetBatchSize();

        int capacity = input_delta->GetCapacity();

        for (int i = 0; i < capacity; i++) {
            (*input_delta)[i] += (*gradient)[i] / batchsize;
        }

        return NULL;
    }

    inline DTYPE Error(DTYPE pred, DTYPE ans) {
        return (pred - ans) * (pred - ans) / 2;
    }
};

#endif  // MSE_H_
