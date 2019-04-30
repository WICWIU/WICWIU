#ifndef WGANGENERATORLOSS_H_
#define WGANGENERATORLOSS_H_    value

#include "../LossFunction.hpp"

template<typename DTYPE>
class WGANGeneratorLoss : public LossFunction<DTYPE>{
public:
    WGANGeneratorLoss(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName){
        #ifdef __DEBUG__
        std::cout << "WGANGeneratorLoss::WGANGeneratorLoss(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator);
    }

    virtual ~WGANGeneratorLoss(){
        #ifdef __DEBUG__
        std::cout << "WGANGeneratorLoss::~WGANGeneratorLoss()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    virtual int Alloc(Operator<DTYPE> *pOperator){
        #ifdef __DEBUG__
        std::cout << "WGANGeneratorLoss::Alloc(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();


        this->SetResult(new Tensor<DTYPE>(timesize, 1, 1, 1, 1));

        return TRUE;
    }

    void Delete(){}

    Tensor<DTYPE>* ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetTensor();
        Tensor<DTYPE> *label  = this->GetLabel()->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize = input->GetRowSize();
        int colsize = input->GetColSize();

        int capacity = channelsize * rowsize * colsize;

        int ti = pTime;


        int start = 0;
        int end   = 0;
        float sumOfLossBatches = 0.f;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                sumOfLossBatches += - (*input)[i];
            }
        }
        if(batchsize != 0)
            (*result)[0] = sumOfLossBatches / batchsize;

        // std::cout << ", Loss Forward " << (*result)[0] << "\n";

        return result;
    }


    Tensor<DTYPE>* BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *input       = this->GetTensor();
        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        int batchsize = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        int start = 0;
        int end   = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                (*input_delta)[i] += - 1.f ;
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

#endif
};

#endif
