#ifndef VANILLAGANDISCRIMINATORLOSS_H_
#define VANILLAGANDISCRIMINATORLOSS_H_    value

#include "../LossFunction.hpp"

template<typename DTYPE>
class VanillaGANDiscriminatorLoss : public LossFunction<DTYPE>{
private:
    DTYPE m_epsilon;
public:
    VanillaGANDiscriminatorLoss(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, DTYPE epsilon, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName){
        #ifdef __DEBUG__
        std::cout << "VanillaGANDiscriminatorLoss::VanillaGANDiscriminatorLoss(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator, epsilon);
    }

    VanillaGANDiscriminatorLoss(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName){
        #ifdef __DEBUG__
        std::cout << "VanillaGANDiscriminatorLoss::VanillaGANDiscriminatorLoss(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator, 1e-6f);
    }

    virtual ~VanillaGANDiscriminatorLoss(){
        #ifdef __DEBUG__
        std::cout << "VanillaGANDiscriminatorLoss::~VanillaGANDiscriminatorLoss()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }

    virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon){
        #ifdef __DEBUG__
        std::cout << "VanillaGANDiscriminatorLoss::Alloc(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();

        this->SetResult(new Tensor<DTYPE>(timesize, 1, 1, 1, 1));

        m_epsilon = epsilon;

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

        // generator를 넣은 D의 계산
        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                // -mean( log(D(x)) - log(D(G(z))) )
                // For each, -log(D(x)) + log(D(G(z)))
                // Label = +1 --> Real input for D, so +1*logD(x)
                // Label = -1 --> Fake input for D, so -1*logD(G(z))
                // Add to result. So result = logD(x) - logD(G(z))
                sumOfLossBatches += - (*label)[i] * log((*input)[i] + m_epsilon) - (1 - (*label)[i]) * log(1.0 - (*input)[i] + m_epsilon);
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
                // ( - 1.0 / D(x) ) + ( 1.0 / D(G(z)) )
                // Label = +1 --> Real input for D, so +1 * logD(x)
                // Label = -1 --> Fake input for D, so -1 * logD(G(z))
                // Add to result. - 넣은 이유는, 위의 식을 따라가기 위함
                (*input_delta)[i] += -((*label)[i] / ((*input)[i] + m_epsilon) - (1 - (*label)[i]) / (1.0 - ((*input)[i] + m_epsilon)));
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
