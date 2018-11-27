#ifndef SOFTMAX_H_
#define SOFTMAX_H_    value

#include "../Operator.hpp"

template<typename DTYPE>
class Softmax : public Operator<DTYPE>{
    DTYPE m_epsilon;

    int m_timesize;

    DTYPE **sum;
    DTYPE **max;

public:
    Softmax(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f) : Operator<DTYPE>(pOperator) {
        #ifdef __DEBUG__
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
        #endif  // __DEBUG__
        Alloc(pOperator, epsilon);
    }

    Softmax(Operator<DTYPE> *pOperator, std::string pName) : Operator<DTYPE>(pOperator, pName) {
        #ifdef __DEBUG__
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
        #endif  // __DEBUG__
        Alloc(pOperator);
    }

    Softmax(Operator<DTYPE> *pOperator, DTYPE epsilon, std::string pName) : Operator<DTYPE>(pOperator, pName) {
        #ifdef __DEBUG__
        std::cout << "Softmax::Softmax(Operator *)" << '\n';
        #endif  // __DEBUG__
        Alloc(pOperator, epsilon);
    }

    ~Softmax() {
        #ifdef __DEBUG__
        std::cout << "Softmax::~Softmax()" << '\n';
        #endif  // __DEBUG__
    }

    virtual int Alloc(Operator<DTYPE> *pOperator, DTYPE epsilon = 1e-6f) {
        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        m_timesize = timesize;

        sum = new DTYPE *[timesize];
        max = new DTYPE *[timesize];

        for (int i = 0; i < timesize; i++) {
            sum[i] = new DTYPE[batchsize];
            max[i] = new DTYPE[batchsize];
        }

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));
        this->SetGradient(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    virtual void Delete() {
        if (sum) {
            for (int i = 0; i < m_timesize; i++) {
                delete[] sum[i];
                sum[i] = NULL;
            }
            delete[] sum;
        }

        if (max) {
            for (int i = 0; i < m_timesize; i++) {
                delete[] max[i];
                max[i] = NULL;
            }
            delete[] max;
        }
    }

    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {  // thread
            sum[ti][ba] = 0.f;
            max[ti][ba] = 0.f;
        }

        int capacity = colsize;

        int start = 0;
        int end   = 0;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            max[ti][ba] = Max(input, start, end);
        }

        DTYPE temp = 0.f;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                temp += (exp((*input)[i] - max[ti][ba]) + m_epsilon);
            }
            sum[ti][ba] = temp;
            temp        = 0.f;
        }

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                (*result)[i] = (exp((*input)[i] - max[ti][ba]) + m_epsilon) / sum[ti][ba];
            }
        }

        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int ti = pTime;

        int capacity = colsize;

        int start = 0;
        int end   = 0;

        float temp = 0.f;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            temp = 0.f;

            for (int i = start; i < end; i++) {
                temp += (*this_delta)[i] * (*result)[i];
            }

            for (int i = start; i < end; i++) {
                (*input_delta)[i] = (*result)[i] * ((*this_delta)[i] - temp);
            }
        }

        return TRUE;
    }

    DTYPE Max(Tensor<DTYPE> *input, int start, int end) {
        DTYPE max = (*input)[start];

        for (int i = start + 1; i < end; i++) {
            if ((*input)[i] > max) max = (*input)[i];
        }

        return max;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {
        this->ForwardPropagate(pTime);
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0) {
        this->BackPropagate(pTime);
        return TRUE;
    }

#endif  // if __CUDNN__
};

#endif  // SOFTMAX_H_
