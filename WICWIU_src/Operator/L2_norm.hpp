#ifndef L2NORM_H_
#define L2NORM_H_    value

#include "../Operator.hpp"

template<typename DTYPE>
class L2_norm : public Operator<DTYPE>{
public:
    L2_norm(Operator<DTYPE> *pInput, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "L2_norm::L2_norm(Operator *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput);
    }


    ~L2_norm() {
        std::cout << "L2_norm::~L2_norm()" << '\n';
    }


    int Alloc(Operator<DTYPE> *pInput) {
        #ifdef __DEBUG__
        std::cout << "L2_norm::Alloc(Operator *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();


        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        return TRUE;
    }


    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_container = this->GetInputContainer();
        Tensor<DTYPE> *input  = (*input_container)[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = input->GetTimeSize();
        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int ba = 0, i = 0; ba < batchsize; ba++) {
            i = ti * batchsize + ba;

            for (int j = 0, index = 0; j < capacity; j++) {
                index         = i * capacity + j;
                  (*result)[i] += ((*input)[index] * (*input)[index]);
                  // (*result)[i] = std::sqrt((*result)[i]);
            }

            // int a;
            // std::cin >> a;
        }
        std::cout << "1" << '\n';
        std::cout << "input" << input<< '\n';
        std::cout << "result" << result<< '\n';
        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_container = this->GetInputContainer();
       //
       Tensor<DTYPE> *input       = (*input_container)[0]->GetResult();
       Tensor<DTYPE> *input_delta = (*input_container)[0]->GetGradient();

        int timesize    = input->GetTimeSize();
        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int ba = 0, i = 0; ba < batchsize; ba++) {
            i = ti * batchsize + ba;

            for (int j = 0, index = 0; j < capacity; j++) {
                index                  = i * capacity + j;
                (*input_delta)[i] += 2.f * ((*input)[index]);
                // (*input_delta)[i] = 1/2 * ((*input_delta)[i]);
            }

        }
        std::cout << "inputdd" << input<< '\n';
        std::cout << "resulddd" << input_delta<< '\n';
        // std::cout << "/* message */" << '\n';
        return TRUE;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime) {
        this->ForwardPropagate(pTime);
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime) {
        this->BackPropagate(pTime);

        return TRUE;
    }

#endif  // __CUDNN__

};

#endif  // SIGMOID_H_
