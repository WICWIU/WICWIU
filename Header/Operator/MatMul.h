#ifndef MATMUL_H_
#define MATMUL_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class MatMul : public Operator<DTYPE>{
public:
    MatMul(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, std::string pName) : Operator<DTYPE>(pInput, pWeight, pName) {
        std::cout << "MatMul::MatMul(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        this->Alloc(pInput, pWeight);
    }

    ~MatMul() {
        std::cout << "MatMul::~MatMul()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight) {
        std::cout << "MatMul::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pWeight->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    int ForwardPropagate() {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        int hiddensize = input->GetColSize();

        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        Shape *inputTenShape  = input->GetShape();
        Shape *weightTenShape = weight->GetShape();
        Shape *resultTenShape = result->GetShape();

        DTYPE temp = 0.f;

        for (int ti = 0; ti < timesize; ti++) {
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            for (int hid = 0; hid < hiddensize; hid++) {
                                (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                                    += (*input)[Index5D(inputTenShape, ti, ba, ch, ro, hid)]
                                       * (*weight)[Index5D(weightTenShape, 0, 0, 0, hid, co)];
                            }
                        }
                    }
                }
            }
        }

        return TRUE;
    }

    int BackPropagate() {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();

        Tensor<DTYPE> *this_delta      = this->GetDelta();
        Tensor<DTYPE> *input_delta     = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[1]->GetGradient();

        int timesize    = this_delta->GetTimeSize();
        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize     = this_delta->GetRowSize();
        int colsize     = this_delta->GetColSize();
        int hiddensize  = input_delta->GetColSize();

        Shape *inputTenShape  = input->GetShape();
        Shape *weightTenShape = weight->GetShape();
        Shape *resultTenShape = this_delta->GetShape();

        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        for (int ti = 0; ti < timesize; ti++) {
            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            for (int hid = 0; hid < hiddensize; hid++) {
                                input_index  = Index5D(inputTenShape, ti, ba, ch, ro, hid);
                                weight_index = Index5D(weightTenShape, 0, 0, 0, hid, co);
                                result_index = Index5D(resultTenShape, ti, ba, ch, ro, co);

                                (*input_delta)[input_index]      += (*weight)[weight_index] * (*this_delta)[result_index];
                                (*weight_gradient)[weight_index] += (*input)[input_index] * (*this_delta)[result_index];
                            }
                        }
                    }
                }
            }
        }

        return TRUE;
    }
};

template<typename DTYPE> class BroadcastMatMul : public Operator<DTYPE>{
private:
    Shape *m_pResultTenShape;

    int m_timesize;
    int m_batchsize;
    int m_channelsize;
    int m_rowsize;
    int m_hidsize;
    int m_colsize;

    int m_ti;
    int m_ba;
    int m_ch;
    int m_ro;
    int m_hid;
    int m_co;

    int *m_ti_weight;
    int *m_ba_weight;
    int *m_ch_weight;
    int *m_ro_weight;
    int *m_hid_weight;
    int *m_co_weight;

    int m_zero;

public:
    BroadcastMatMul(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight, std::string pName) : Operator<DTYPE>(pInput, pWeight, pName) {
        std::cout << "BroadcastMatMul::BroadcastMatMul(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        this->Alloc(pInput, pWeight);
    }

    ~BroadcastMatMul() {
        std::cout << "BroadcastMatMul::~BroadcastMatMul()" << '\n';
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeight) {
        std::cout << "BroadcastMatMul::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';

        Shape *pInputTenShape  = pInput->GetResult()->GetShape();
        Shape *pWeightTenShape = pWeight->GetResult()->GetShape();

        m_timesize    = (*pInputTenShape)[0];
        m_batchsize   = (*pInputTenShape)[1];
        m_channelsize = (*pInputTenShape)[2];
        m_rowsize     = (*pInputTenShape)[3];
        m_hidsize     = (*pInputTenShape)[4];
        m_colsize     = (*pWeightTenShape)[4];

        m_ti  = 0;
        m_ba  = 0;
        m_ch  = 0;
        m_ro  = 0;
        m_hid = 0;
        m_co  = 0;

        m_ti_weight  = &m_ti;
        m_ba_weight  = &m_ba;
        m_ch_weight  = &m_ch;
        m_ro_weight  = &m_ro;
        m_hid_weight = &m_hid;
        m_co_weight  = &m_co;

        m_zero = 0;

        if ((*pWeightTenShape)[0] == 1) m_ti_weight = &m_zero;

        if ((*pWeightTenShape)[1] == 1) m_ba_weight = &m_zero;

        if ((*pWeightTenShape)[2] == 1) m_ch_weight = &m_zero;

        m_pResultTenShape = new Shape(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize);
        this->SetResult(new Tensor<DTYPE>(m_pResultTenShape));

        this->SetDelta(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        return TRUE;
    }

    int ForwardPropagate() {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *weight = (*input_contatiner)[1]->GetResult();

        Tensor<DTYPE> *result = this->GetResult();

        Shape *inputTenShape  = input->GetShape();
        Shape *weightTenShape = weight->GetShape();

        for (m_ti = 0; m_ti < m_timesize; m_ti++) {
            for (m_ba = 0; m_ba < m_batchsize; m_ba++) {
                for (m_ch = 0; m_ch < m_channelsize; m_ch++) {
                    for (m_ro = 0; m_ro < m_rowsize; m_ro++) {
                        for (m_co = 0; m_co < m_colsize; m_co++) {
                            for (m_hid = 0; m_hid < m_hidsize; m_hid++) {
                                (*result)[Index5D(m_pResultTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                                    += (*input)[Index5D(inputTenShape, m_ti, m_ba, m_ch, m_ro, m_hid)] * (*weight)[Index5D(weightTenShape, *m_ti_weight, *m_ba_weight, *m_ch_weight, *m_hid_weight, *m_co_weight)];
                            }
                        }
                    }
                }
            }
        }

        return TRUE;
    }

    int BackPropagate() {
        // Tensor<DTYPE> *this_delta = Tensor<DTYPE>::Constants(1, 3, 1, 3, 3, 1.0);

        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weight = this->GetInput()[1]->GetResult();

        Tensor<DTYPE> *this_delta      = this->GetDelta();
        Tensor<DTYPE> *input_delta     = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[1]->GetGradient();

        Shape *inputTenShape  = input->GetShape();
        Shape *weightTenShape = weight->GetShape();

        int input_index  = 0;
        int weight_index = 0;
        int result_index = 0;

        for (m_ti = 0; m_ti < m_timesize; m_ti++) {
            for (m_ba = 0; m_ba < m_batchsize; m_ba++) {
                for (m_ch = 0; m_ch < m_channelsize; m_ch++) {
                    for (m_ro = 0; m_ro < m_rowsize; m_ro++) {
                        for (m_co = 0; m_co < m_colsize; m_co++) {
                            for (m_hid = 0; m_hid < m_hidsize; m_hid++) {
                                input_index  = Index5D(inputTenShape, m_ti, m_ba, m_ch, m_ro, m_hid);
                                weight_index = Index5D(weightTenShape, *m_ti_weight, *m_ba_weight, *m_ch_weight, *m_hid_weight, *m_co_weight);
                                result_index = Index5D(m_pResultTenShape, m_ti, m_ba, m_ch, m_ro, m_co);

                                (*input_delta)[input_index]      += (*weight)[weight_index] * (*this_delta)[result_index];
                                (*weight_gradient)[weight_index] += (*input)[input_index] * (*this_delta)[result_index];
                            }
                        }
                    }
                }
            }
        }

        return TRUE;
    }
};


#endif  // MATMUL_H_
