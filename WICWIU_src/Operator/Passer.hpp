#ifndef PASSER_H_
#define PASSER_H_    value

#include "../Operator.hpp"

template<typename DTYPE>
class Passer : public Operator<DTYPE>{
private:
    int m_noOperator;

public:
    Passer(Operator<DTYPE> *anc, Operator<DTYPE> *pos, Operator<DTYPE> *neg, 
            Operator<DTYPE> *neg2, std::string pName = "NO NAME", int pLoadflag = TRUE) 
            : Operator<DTYPE>(4, anc, pos, neg, neg2, pLoadflag) {
        this->Alloc(4, anc, pos, neg, neg2);
    }

    ~Passer() { }

    int Alloc(int noOperator, ...) {
        m_noOperator  = noOperator;
        va_list ap;
        va_start(ap, noOperator);

        Operator<DTYPE> *temp = va_arg(ap, Operator<DTYPE> *);
        
        int timesize    = temp->GetResult()->GetTimeSize();
        int batchsize   = temp->GetResult()->GetBatchSize();
        int channelsize = temp->GetResult()->GetChannelSize();
        int rowsize     = temp->GetResult()->GetRowSize();
        int colsize     = temp->GetResult()->GetColSize();

        va_end(ap);
        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, m_noOperator, rowsize, colsize));
        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, m_noOperator, rowsize, colsize));
        return TRUE;
    }

    int ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = NULL;
        Tensor<DTYPE> *result = this->GetResult();

        int timesize  = result->GetTimeSize();
        int batchsize = result->GetBatchSize();
        int rowsize   = result->GetRowSize();
        int colsize   = result->GetColSize();

        Shape *inputTenShape  = NULL;
        Shape *resultTenShape = result->GetShape();

        int ti = pTime;

        for (int i = 0; i < m_noOperator; i++) {
            input         = this->GetInput()[i]->GetResult();
            inputTenShape = input->GetShape();
            int channelsize = input->GetChannelSize();

            for (int ba = 0; ba < batchsize; ba++) 
            for (int ch = 0; ch < channelsize; ch++) 
            for (int ro = 0; ro < rowsize; ro++)
            for (int co = 0; co < colsize; co++) 
                (*result)[Index5D(resultTenShape, ti, ba, i, ro, co)]
                    = (*input)[Index5D(inputTenShape, ti, ba, ch, ro, co)];
        }

        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = NULL;

        int timesize  = this_delta->GetTimeSize();
        int batchsize = this_delta->GetBatchSize();
        int rowsize   = this_delta->GetRowSize();
        int colsize   = this_delta->GetColSize();

        Shape *inputTenShape  = NULL;
        Shape *resultTenShape = this_delta->GetShape();

        int ti = pTime;

        for (int i = 0; i < m_noOperator; i++) {
            input_delta   = this->GetInput()[i]->GetDelta();
            inputTenShape = input_delta->GetShape();
            int channelsize = input_delta->GetChannelSize();

            for (int ba = 0; ba < batchsize; ba++) 
            for (int ch = 0; ch < channelsize; ch++) 
            for (int ro = 0; ro < rowsize; ro++)
            for (int co = 0; co < colsize; co++) 
                (*input_delta)[Index5D(inputTenShape, ti, ba, ch, ro, co)]
                    += (*this_delta)[Index5D(resultTenShape, ti, ba, i, ro, co)];
        }

        return TRUE;
    }

 #ifdef __CUDNN__
     int ForwardPropagateOnGPU(int pTime);
     int BackPropagateOnGPU(int pTime);
 #endif  // __CUDNN__
};

#endif  // PASSER_H_
