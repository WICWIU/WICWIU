#ifndef ADD_H_
#define ADD_H_    value

#include "..//Operator.h"

template<typename DTYPE> class Addall : public Operator<DTYPE>{
private:
    int m_capacity;

public:
    Addall(Operator<DTYPE> *pLeftInput, Operator<DTYPE> *pRightInput, std::string pName) : Operator<DTYPE>(pLeftInput, pRightInput, pName) {
        std::cout << "Addall::Addall(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        this->Alloc(pLeftInput, pRightInput);
    }

    ~Addall() {
        std::cout << "Addall::~Addall()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pLeftInput, Operator<DTYPE> *pRightInput) {
        std::cout << "Addall::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        Shape *pInputTenShape = pLeftInput->GetResult()->GetShape();

        int timesize    = (*pInputTenShape)[0];
        int batchsize   = (*pInputTenShape)[1];
        int channelsize = (*pInputTenShape)[2];
        int rowsize     = (*pInputTenShape)[3];
        int colsize     = (*pInputTenShape)[4];

        m_capacity = pLeftInput->GetResult()->GetCapacity();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetGradient(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    int ForwardPropagate() {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *left_input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *right_input = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *result      = this->GetResult();

        for (int i = 0; i < m_capacity; i++) {
            (*result)[i] = (*left_input)[i] + (*right_input)[i];
        }

        return TRUE;
    }

    int BackPropagate() {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *left_input_grad  = (*input_contatiner)[0]->GetGradient();
        Tensor<DTYPE> *right_input_grad = (*input_contatiner)[1]->GetGradient();
        Tensor<DTYPE> *this_grad        = this->GetGradient();

        for (int i = 0; i < m_capacity; i++) {
            (*left_input_grad)[i]  += (*this_grad)[i];
            (*right_input_grad)[i] += (*this_grad)[i];
        }

        return TRUE;
    }
};

template<typename DTYPE> class Add : public Operator<DTYPE>{
private:
    Shape *m_pInputTenShape;
    Shape *m_pBiasTenShape;

    int m_timesize;
    int m_batchsize;
    int m_channelsize;
    int m_rowsize;
    int m_colsize;

    int m_ti;
    int m_ba;
    int m_ch;
    int m_ro;
    int m_co;

    int *m_ti_bias;
    int *m_ba_bias;
    int *m_ch_bias;
    int *m_ro_bias;
    int *m_co_bias;

    int m_zero;

public:
    Add(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias, std::string pName) : Operator<DTYPE>(pInput, pBias, pName) {
        std::cout << "Add::Add(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        this->Alloc(pInput, pBias);
    }

    ~Add() {
        std::cout << "Add::~Add()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pBias) {
        std::cout << "Add::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        m_pInputTenShape = pInput->GetResult()->GetShape();
        m_pBiasTenShape  = pBias->GetResult()->GetShape();

        m_timesize    = (*m_pInputTenShape)[0];
        m_batchsize   = (*m_pInputTenShape)[1];
        m_channelsize = (*m_pInputTenShape)[2];
        m_rowsize     = (*m_pInputTenShape)[3];
        m_colsize     = (*m_pInputTenShape)[4];

        m_ti = 0;
        m_ba = 0;
        m_ch = 0;
        m_ro = 0;
        m_co = 0;

        m_ti_bias = &m_ti;
        m_ba_bias = &m_ba;
        m_ch_bias = &m_ch;
        m_ro_bias = &m_ro;
        m_co_bias = &m_co;

        m_zero = 0;

        if ((*m_pBiasTenShape)[0] == 1) m_ti_bias = &m_zero;

        if ((*m_pBiasTenShape)[1] == 1) m_ba_bias = &m_zero;

        if ((*m_pBiasTenShape)[2] == 1) m_ch_bias = &m_zero;

        if ((*m_pBiasTenShape)[3] == 1) m_ro_bias = &m_zero;

        if ((*m_pBiasTenShape)[4] == 1) m_co_bias = &m_zero;

        this->SetResult(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        this->SetGradient(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        return TRUE;
    }

    int ForwardPropagate() {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *bias   = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        for (m_ti = 0; m_ti < m_timesize; m_ti++) {
            for (m_ba = 0; m_ba < m_batchsize; m_ba++) {
                for (m_ch = 0; m_ch < m_channelsize; m_ch++) {
                    for (m_ro = 0; m_ro < m_rowsize; m_ro++) {
                        for (m_co = 0; m_co < m_colsize; m_co++) {
                            (*result)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                                = (*input)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                                  + (*bias)[Index5D(m_pBiasTenShape, *m_ti_bias, *m_ba_bias, *m_ch_bias, *m_ro_bias, *m_co_bias)];
                        }
                    }
                }
            }
        }

        return TRUE;
    }

    int BackPropagate() {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input_grad = (*input_contatiner)[0]->GetGradient();
        Tensor<DTYPE> *bias_grad  = (*input_contatiner)[1]->GetGradient();
        Tensor<DTYPE> *this_grad  = this->GetGradient();

        for (m_ti = 0; m_ti < m_timesize; m_ti++) {
            for (m_ba = 0; m_ba < m_batchsize; m_ba++) {
                for (m_ch = 0; m_ch < m_channelsize; m_ch++) {
                    for (m_ro = 0; m_ro < m_rowsize; m_ro++) {
                        for (m_co = 0; m_co < m_colsize; m_co++) {
                            (*input_grad)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                                += (*this_grad)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)];

                            (*bias_grad)[Index5D(m_pBiasTenShape, *m_ti_bias, *m_ba_bias, *m_ch_bias, *m_ro_bias, *m_co_bias)]
                                += (*this_grad)[Index5D(m_pInputTenShape, m_ti, m_ba, m_ch, m_ro, m_co)];
                        }
                    }
                }
            }
        }

        return TRUE;
    }
};

#endif  // ADD_H_
