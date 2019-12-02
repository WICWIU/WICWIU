#ifndef RECONSTRUCTIONERROR_H_
#define RECONSTRUCTIONERROR_H_    value

#include "../Operator.hpp"

template<typename DTYPE>
class ReconstructionError : public Operator<DTYPE>{
public:
    ReconstructionError(Operator<DTYPE> *pInput, Operator<DTYPE> *pLabel, std::string pName) : Operator<DTYPE>(pInput, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "MSE::MSE(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput);
    }

    virtual ~ReconstructionError() {
        #ifdef __DEBUG__
        std::cout << "MSE::~MSE()" << '\n';
        #endif  // __DEBUG__
    }

    virtual int Alloc(Operator<DTYPE> *pInput) {
        #ifdef __DEBUG__
        std::cout << "MSE::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pInput->GetResult()->GetTimeSize();

        this->SetResult(new Tensor<DTYPE>(timesize, 1, 1, 1, 1));
        this->SetDelta(new Tensor<DTYPE>(timesize, 1, 1, 1, 1));

        return TRUE;
    }

    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_container = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_container)[0]->GetResult();
        Tensor<DTYPE> *label  = (*input_container)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = batchsize * channelsize * rowsize * colsize;

        int ti = pTime;

        for (int j = 0, index = 0; j < capacity; j++) {
            index         = ti * capacity + j;
            (*result)[ti] += std::abs((*input)[index] - (*label)[index]);
            // if(index < 784)
                // std::cout << "index =  " << index << ", input = " << (*input)[index] << ", label = " << (*label)[index] << ", result = " << std::abs((*input)[index] - (*label)[index]) << "\n";
        }
        (*result)[ti] = (*result)[ti] / capacity;
        // std::cout <<"totalResult = " << (*result)[ti] << "\n";

        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_container = this->GetInputContainer();

        Tensor<DTYPE> *input       = (*input_container)[0]->GetResult();
        Tensor<DTYPE> *label       = (*input_container)[1]->GetResult();
        Tensor<DTYPE> *input_delta = (*input_container)[0]->GetGradient();
        Tensor<DTYPE> *label_delta = (*input_container)[1]->GetGradient();
        Tensor<DTYPE> *this_delta  = this->GetGradient();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = batchsize * channelsize * rowsize * colsize;
        int divisor     = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int j = 0, index = 0; j < capacity; j++) {
            index                 = ti * capacity + j;
            if((*input)[index] - (*label)[index] > 0){
                (*input_delta)[index] += (*this_delta)[ti] / divisor;
                (*label_delta)[index] += -(*this_delta)[ti] / divisor;
            }
            else if ((*input)[index] - (*label)[index] < 0) {
                (*input_delta)[index] += -(*this_delta)[ti] / divisor;
                (*label_delta)[index] += (*this_delta)[ti] / divisor;
            }
            // else{
            //     (*input_delta)[index] += 0;
            // }
	    /*if((index < 784)&&(index % 100 == 0)){
	    	std::cout << "index = " << index <<  ", (*input)[index] = " << ((*input)[index] + 1) / 2 * 255 << ", (*label)[index] = " << ((*label)[index] + 1) / 2 * 255 << ", (*input)[index] - (*label)[index] = " << (*input)[index] - (*label)[index] << ", (*this_delta)[ti] = " << (*this_delta)[ti] << ", (*input_delta)[index] = " << (*input_delta)[index] << ", (*label_delta)[index] = " <<(*label_delta)[index] << "\n";
	     }*/
	}
        return TRUE;
    }

#ifdef __CUDNN__

    int ForwardPropagateOnGPU(int pTime = 0) {
        this->ForwardPropagate();
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0) {
        this->BackPropagate();
        return TRUE;
    }

#endif  // __CUDNN__

};

#endif  // L1NORM_H_
