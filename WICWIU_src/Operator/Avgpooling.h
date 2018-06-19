#ifndef __AVGPOOLING__
#define __AVGPOOLING__    value

#include "../Operator.h"

template<typename DTYPE> class GlobalAvaragePooling2D : public Operator<DTYPE>{
private:
    int m_timesize;
    int m_batchsize;
    int m_channelsize;
    int m_rowsize;
    int m_colsize;

    int m_divisor;

public:
    GlobalAvaragePooling2D(Operator<DTYPE> *pInput, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "GlobalAvaragePooling2D::GlobalAvaragePooling2D(Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        Alloc(pInput);
    }

    virtual ~GlobalAvaragePooling2D() {}

    int Alloc(Operator<DTYPE> *pInput) {
        Shape *pInputTenShape = pInput->GetResult()->GetShape();

        m_timesize    = (*pInputTenShape)[0];
        m_batchsize   = (*pInputTenShape)[1];
        m_channelsize = (*pInputTenShape)[2];
        m_rowsize     = (*pInputTenShape)[3];
        m_colsize     = (*pInputTenShape)[4];

        m_divisor = m_rowsize * m_colsize;

        this->AddResult(new Tensor<DTYPE>(new Shape(m_timesize, m_batchsize, m_channelsize, 1, 1)));
        this->AddGradient(new Tensor<DTYPE>(new Shape(m_timesize, m_batchsize, m_channelsize, 1, 1)));

        return TRUE;
    }

    int ForwardPropagate(int pTime = 0, int pThreadNum = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = result->GetShape();

        int ti = pTime;
        int numOfThread = this->GetNumOfThread();

        for (int ba = pThreadNum; ba < m_batchsize; ba += numOfThread) {
            for (int ch = 0; ch < m_channelsize; ch++) {
                for (int ro = 0; ro < m_rowsize; ro++) {
                    for (int co = 0; co < m_colsize; co++) {
                        (*result)[Index5D(resultTenShape, ti, ba, ch, 0, 0)]
                            += (*input)[Index5D(inputTenShape, ti, ba, ch, ro, co)];
                    }
                }
                (*result)[Index5D(resultTenShape, ti, ba, ch, 0, 0)] /= m_divisor;
            }
        }


        return TRUE;
    }

    int BackPropagate(int pTime = 0, int pThreadNum = 0) {
        Container<Operator<DTYPE> *> *input_contatiner         = this->GetInputContainer();
        Container<Tensor<DTYPE> *>   *input_gradient_container = (*input_contatiner)[0]->GetGradientContainer();
        Container<Tensor<DTYPE> *>   *this_gradient_container  = this->GetGradientContainer();

        Tensor<DTYPE> *this_grad  = (*this_gradient_container)[0];
        Tensor<DTYPE> *input_grad = (*input_gradient_container)[0];

        Shape *resultTenShape = this_grad->GetShape();
        Shape *inputTenShape  = input_grad->GetShape();

        int ti = pTime;
        int numOfThread = this->GetNumOfThread();

        for (int ba = pThreadNum; ba < m_batchsize; ba += numOfThread) {
            for (int ch = 0; ch < m_channelsize; ch++) {
                for (int ro = 0; ro < m_rowsize; ro++) {
                    for (int co = 0; co < m_colsize; co++) {
                        (*input_grad)[Index5D(inputTenShape, ti, ba, ch, ro, co)]
                            += (*this_grad)[Index5D(resultTenShape, ti, ba, ch, 0, 0)] / m_divisor;
                    }
                }
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
//
#endif  // __AVGPOOLING__
