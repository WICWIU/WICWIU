#ifndef L2NORM_H_
#define L2NORM_H_    value

#include "../Operator.hpp"

template<typename DTYPE>
class L2_normalize : public Operator<DTYPE>{
private:
  DTYPE m_epsilon;
public:
    L2_normalize(Operator<DTYPE> *pInput, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "L2_norm::L2_norm(Operator *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput);
    }


    ~L2_normalize() {
        std::cout << "L2_normalize::~L2_normalize()" << '\n';
    }


    int Alloc(Operator<DTYPE> *pInput) {
        #ifdef __DEBUG__
        std::cout << "L2_normalize::Alloc(Operator *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int colsize   = pInput->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, 1, 1, colsize));

        m_epsilon = 1e-12;

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
        DTYPE sqrt_x = 0.f;

        for (int ba = 0, i = 0; ba < batchsize; ba++) {
            i = ti * batchsize + ba;

            for (int j = 0, index = 0; j < capacity; j++) {
                index         = i * capacity + j;
                (*result)[i] += ((*input)[index] * (*input)[index]);

            }

            sqrt_x = std::sqrt((*result)[i]);

            for (int j = 0, index = 0; j < capacity; j++) {
                index         = i * capacity + j;
                (*result)[index] = ((*input)[index] / (sqrt_x + m_epsilon));
            }
        }

        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
      Container<Operator<DTYPE> *> *input_container = this->GetInputContainer();
       //
       Tensor<DTYPE> *input       = (*input_container)[0]->GetResult();
       Tensor<DTYPE> *result      = this->GetResult();
       Tensor<DTYPE> *input_delta = (*input_container)[0]->GetGradient();
       Tensor<DTYPE> *this_delta  = this->GetDelta();

        int timesize    = this_delta->GetTimeSize();
        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize     = this_delta->GetRowSize();
        int colsize     = this_delta->GetColSize();

        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int ba = 0, i = 0; ba < batchsize; ba++) {
            i = ti * batchsize + ba;

            for (int j = 0, index = 0; j < capacity; j++) {
                index                  = i * capacity + j;

                if(i == index)
                  (*input_delta)[index] += ((((*result)[i] - (*input)[index]) / (std::pow((*result)[i], 1.5) + m_epsilon)) * (*this_delta)[index]);
                else
                  (*input_delta)[index] -= ((((*input)[index] *(*input)[i]) / (std::pow((*result)[i], 1.5) + m_epsilon)) * (*this_delta)[index]);
            }
        }

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
