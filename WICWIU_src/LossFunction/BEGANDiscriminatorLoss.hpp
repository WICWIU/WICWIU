#ifndef BEGANDISCRIMINATORLOSS_H_
#define BEGANDISCRIMINATORLOSS_H_    value

#include "../LossFunction.hpp"

template<typename DTYPE>
class BEGANDiscriminatorLoss : public LossFunction<DTYPE>{
public:
    BEGANDiscriminatorLoss(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName){
        #ifdef __DEBUG__
        std::cout << "BEGANDiscriminatorLoss::BEGANDiscriminatorLoss(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator);
    }

    virtual ~BEGANDiscriminatorLoss(){
        #ifdef __DEBUG__
        std::cout << "BEGANDiscriminatorLoss::~BEGANDiscriminatorLoss()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    virtual int Alloc(Operator<DTYPE> *pOperator){
        #ifdef __DEBUG__
        std::cout << "BEGANDiscriminatorLoss::Alloc(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        return TRUE;
    }

    void Delete(){}

    Tensor<DTYPE>* ForwardPropagate(int pTime = 0){
        Tensor<DTYPE> *input = this->GetTensor();
        Tensor<DTYPE> *label = this->GetLabel()->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity = channelsize * rowsize * colsize;
        int ti = pTime;


        int start = 0;
        int end   = 0;
        float sumOfLossBatches = 0.f;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                // -mean( D(x) - D(G(z)) )
                // Label = 1 --> Real input for D, D(x)
                // Label = 0 --> Fake input for D, -D(G(z))
                // Add to result. So result = - label * D(x) + ( 1 - label ) * D(G(z))
                // Global optim for D_loss = 0
                sumOfLossBatches += (*label)[i] * (*input)[i] - (1.f - (*label)[i]) * (*input)[i];
            }
        }
        // std::cout << "(*label)[i] : " << (*label)[0] << '\n';
        // std::cout << "(*input)[0] : " << (*input)[0] << ", ";
        // std::cout << "(*label)[i] * log((*input)[i] + m_epsilon) : " << (*label)[0] * log((*input)[0] + m_epsilon) << '\n';
        // std::cout << "(1 - (*label)[i]) * log(1.0 - (*input)[i] + m_epsilon) : " << (1 - (*label)[0]) * log(1.0 - (*input)[0] + m_epsilon) << '\n';
        // std::cout << "(*label)[i] * log((*input)[i] + m_epsilon) + (1 - (*label)[i]) * log(1.0 - (*input)[i] + m_epsilon) : " << (*label)[0] * log((*input)[0] + m_epsilon) + (1 - (*label)[0]) * log(1.0 - (*input)[0] + m_epsilon) << '\n';
        // std::cout << "sumOfLossBatches : " << sumOfLossBatches << '\n';


        if(batchsize != 0)
            (*result)[0] += sumOfLossBatches / batchsize;
        // std::cout << ", Loss Forward " << (*result)[0] << "\n";

        return result;
    }


    Tensor<DTYPE>* BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *input       = this->GetTensor();
        Tensor<DTYPE> *label       = this->GetLabel()->GetResult();
        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        int batchsize = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize   = input->GetRowSize();
        int colsize   = input->GetColSize();
        int capacity  = channelsize * rowsize * colsize;

        int start = 0;
        int end   = 0;

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                // for Real = 1, for Fake = -1
                // Label = 1
                // Label = 0
                // Add to result. - label + ( 1 - label )
                (*input_delta)[i] += (*label)[i] - (1.f - (*label)[i]);
            }

        }

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
