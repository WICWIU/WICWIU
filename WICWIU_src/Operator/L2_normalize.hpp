#ifndef L2NORM_H_
#define L2NORM_H_    value

#include "../Operator.hpp"

template<typename DTYPE>
class L2_normalize : public Operator<DTYPE>{
private:
  DTYPE m_eplsilon;
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

        m_eplsilon = 1e-08;

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
                if((*result)[i] > m_eplsilon)
                  (*result)[index] = ((*input)[index] / std::sqrt((*result)[i]));
                else (*result)[index] = ((*input)[index] / std::sqrt(m_eplsilon));

            }

        }

        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
      Container<Operator<DTYPE> *> *input_container = this->GetInputContainer();
       //
       Tensor<DTYPE> *input       = (*input_container)[0]->GetResult();
       Tensor<DTYPE> *input_delta = (*input_container)[0]->GetGradient();
       Tensor<DTYPE> *this_delta  = this->GetDelta();

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
                (*input_delta)[index] += (*this_delta)[index];
            }

        }

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
