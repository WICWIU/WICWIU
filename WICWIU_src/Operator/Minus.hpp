//#ifndef MINUS_H_
//#define MINUS_H_    value

#include "../Operator.hpp"


template<typename DTYPE> class Minus : public Operator<DTYPE>{
private:
    Shape *m_pLeftTenShape;
    ///< 연산 할 Tensor의 Shape
    Shape *m_pRightTenShape;
    ///< 연산 할 Tensor의 Shape

    int m_timeSize;
    ///< time의 dimension 크기
    int m_batchSize;
    ///< batch의 dimension 크기
    int m_channelSize;
    ///< channel의 dimension 크기
    int m_rowSize;
    ///< row의 dimension 크기
    int m_colSize;
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

        m_timeSize    = (*m_pLeftTenShape)[0];
        m_batchSize   = (*m_pLeftTenShape)[1];
        m_channelSize = (*m_pLeftTenShape)[2];
        m_rowSize     = (*m_pLeftTenShape)[3];
        m_colSize     = (*m_pLeftTenShape)[4];

        this->SetResult(new Tensor<DTYPE>(m_timeSize, m_batchSize, m_channelSize, m_rowSize, m_colSize));

        this->SetGradient(new Tensor<DTYPE>(m_timeSize, m_batchSize, m_channelSize, m_rowSize, m_colSize));

        return TRUE;
    }

    void Delete() {}

    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *inputContatiner = this->GetInputContainer();

        Tensor<DTYPE> *left   = (*inputContatiner)[0]->GetResult();
        Tensor<DTYPE> *right  = (*inputContatiner)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int m_ti = pTime;

        for (int m_ba = 0; m_ba < m_batchSize; m_ba++) {
            for (int m_ch = 0; m_ch < m_channelSize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowSize; m_ro++) {
                    for (int m_co = 0; m_co < m_colSize; m_co++) {
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
        Container<Operator<DTYPE> *> *inputContatiner = this->GetInputContainer();

        Tensor<DTYPE> *left_grad  = (*inputContatiner)[0]->GetGradient();
        Tensor<DTYPE> *right_grad = (*inputContatiner)[1]->GetGradient();
        Tensor<DTYPE> *this_grad  = this->GetGradient();

        int m_ti = pTime;

        for (int m_ba = 0; m_ba < m_batchSize; m_ba++) {
            for (int m_ch = 0; m_ch < m_channelSize; m_ch++) {
                for (int m_ro = 0; m_ro < m_rowSize; m_ro++) {
                    for (int m_co = 0; m_co < m_colSize; m_co++) {

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
