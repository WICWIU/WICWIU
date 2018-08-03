#ifndef HINGELOSS_H_
#define HINGELOSS_H_    value

#include "../LossFunction.h"

template<typename DTYPE>
class HingeLoss : public LossFunction<DTYPE>{
private:
    Tensor<DTYPE> *indexForBackProp;

public:
    HingeLoss(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel) : LossFunction<DTYPE>(pOperator, pLabel) {
        #ifdef __DEBUG__
        std::cout << "HingeLoss::HingeLoss(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator);
    }

    HingeLoss(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "HingeLoss::HingeLoss(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator);
    }

    ~HingeLoss() {
        #ifdef __DEBUG__
        std::cout << "HingeLoss::~HingeLoss()" << '\n';
        #endif  // __DEBUG__
    }

    virtual int Alloc(Operator<DTYPE> *pOperator) {
        #ifdef __DEBUG__
        std::cout << "HingeLoss::Alloc(Operator<DTYPE> *, Operator<DTYPE> *, int)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchSize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchSize, 1, 1, 1));

        indexForBackProp = Tensor<DTYPE>::Constants(timesize, batchSize, channelsize, rowsize, colsize, 0.f);

        return TRUE;
    }

    Tensor<DTYPE>* ForwardPropagate(int timeIdx = 0) {
        Tensor<DTYPE> *input   = this->GetTensor();
        Tensor<DTYPE> *desired = this->GetLabel()->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();
        indexForBackProp->Reset();

        int batchSize = input->GetBatchSize();

        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int outputDim   = channelsize * rowsize * colsize;

        int   trueClass      = 0;
        DTYPE trueClassScore = (DTYPE)0;

        for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++) {
            int globalSampleIdx = (timeIdx * batchSize + sampleIdx);

            DTYPE *dStart = &(*desired)[globalSampleIdx * outputDim];
            DTYPE *dLimit = dStart + outputDim;

            for (DTYPE *dp = dStart; dp < dLimit; dp++) {
                if (*dp == (DTYPE)1) {
                    trueClass = dp - dStart;
                    break;
                }
            }

            int firstOutputIdx = globalSampleIdx * outputDim;
            trueClassScore = (*input)[firstOutputIdx + trueClass];

            DTYPE *rp       = &(*result)[globalSampleIdx];    // output
            DTYPE *ip       = &(*input)[firstOutputIdx];      // input index
            int    curClass = 0;

            for (DTYPE *dp = dStart; dp < dLimit; dp++, curClass++, ip++) {
                if ((*ip + 1 > trueClassScore) && (curClass != trueClass)) {
                    *rp += *ip - trueClassScore + 1;

                    // for backpropagation, not necessary for forward
                    (*indexForBackProp)[curClass + firstOutputIdx]  += 1;
                    (*indexForBackProp)[trueClass + firstOutputIdx] += -1;
                }
            }
        }

        return result;
    }

    Tensor<DTYPE>* BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *input       = this->GetTensor();
        Tensor<DTYPE> *label       = this->GetLabel()->GetResult();
        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        int batchSize = input->GetBatchSize();

        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        int temp = 0;

        for (int ba = 0, i = 0; ba < batchSize; ba++) {
            i = ti * batchSize + ba;

            for (int j = 0, index = 0; j < capacity; j++) {
                index = i * capacity + j;

                (*input_delta)[index] += (*indexForBackProp)[index] / batchSize;
            }
        }


        return NULL;
    }

#ifdef __CUDNN__

    Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0) {
        this->ForwardPropagate();
        return NULL;
    }

    Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0) {
        this->BackPropagate();
        return NULL;
    }

#endif  // __CUDNN__
};

#endif  // HINGELOSS_H_
