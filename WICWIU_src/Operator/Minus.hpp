//#ifndef MINUS_H_
//#define MINUS_H_    value

#include "../Operator.hpp"


template<typename DTYPE> class Minus : public Operator<DTYPE>{
private:
    Shape *m_pLeftTenShape;
    ///< 연산 할 Tensor의 Shape
    Shape *m_pRightTenShape;
    ///< 연산 할 Tensor의 Shape

    int m_timesize;
    ///< time의 dimension 크기
    int m_batchsize;
    ///< batch의 dimension 크기
    int m_channelsize;
    ///< channel의 dimension 크기
    int m_rowsize;
    ///< row의 dimension 크기
    int m_colsize;
    ///< col의 dimension 크기

public:
    Minus(Operator<DTYPE> *pLeftInput, Operator<DTYPE> *pRightInput, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pLeftInput, pRightInput, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "Minusall::Minusall(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pLeftInput, pRightInput);
    }


    ~Minus() {
        #ifdef __DEBUG__
        std::cout << "Minusall::~Minusall()" << '\n';
        #endif  // __DEBUG__
    }

    int Alloc(Operator<DTYPE> *pLeftInput, Operator<DTYPE> *pRightInput) {
        #ifdef __DEBUG__
        std::cout << "Minusall::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        m_pLeftTenShape  = pLeftInput->GetResult()->GetShape();
        m_pRightTenShape = pRightInput->GetResult()->GetShape();

        m_timesize    = (*m_pLeftTenShape)[0];
        m_batchsize   = (*m_pLeftTenShape)[1];
        m_channelsize = (*m_pLeftTenShape)[2];
        m_rowsize     = (*m_pLeftTenShape)[3];
        m_colsize     = (*m_pLeftTenShape)[4];

        this->SetResult(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        this->SetGradient(new Tensor<DTYPE>(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize));

        return TRUE;
    }

    void Delete() {}

    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *left   = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *right  = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int m_ti = pTime;

        for (int m_ba = 0; m_ba < m_batchsize; m_ba++) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {
                        (*result)[Index5D(m_pLeftTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                            = (*left)[Index5D(m_pLeftTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                              - (*right)[Index5D(m_pRightTenShape, m_ti, m_ba, m_ch, m_ro, m_co)];
                    }
                }
            }
        }

        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *left_grad  = (*input_contatiner)[0]->GetGradient();
        Tensor<DTYPE> *right_grad = (*input_contatiner)[1]->GetGradient();
        Tensor<DTYPE> *this_grad  = this->GetGradient();

        int m_ti = pTime;

        for (int m_ba = 0; m_ba < m_batchsize; m_ba++) {
            for (int m_ch = 0; m_ch < m_channelsize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowsize; m_ro++) {
                    for (int m_co = 0; m_co < m_colsize; m_co++) {

                        (*left_grad)[Index5D(m_pLeftTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                            += (*this_grad)[Index5D(m_pLeftTenShape, m_ti, m_ba, m_ch, m_ro, m_co)];

                        (*right_grad)[Index5D(m_pRightTenShape, m_ti, m_ba, m_ch, m_ro, m_co)]
                            -= (*this_grad)[Index5D(m_pLeftTenShape, m_ti, m_ba, m_ch, m_ro, m_co)];
                    }
                }
            }
        }
        return TRUE;
    }
};


//#endif  // MINUS_H_
