#ifndef CONCATENATECHANNELWISE_H_
#define CONCATENATECHANNELWISE_H_    value

#include "../Operator.h"

template<typename DTYPE>
class ConcatenateChannelWise : public Operator<DTYPE>{
private:
    int m_noOperator;

public:
    ConcatenateChannelWise(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName = "NO NAME") : Operator<DTYPE>(pInput0, pInput1, pName) {
        #ifdef __DEBUG__
        std::cout << "ConcatenateChannelWise::ConcatenateChannelWise(Operator *)" << '\n';
        #endif  // __DEBUG__

        m_noOperator = 0;
        this->Alloc(2, pInput0, pInput1);
    }

    ~ConcatenateChannelWise() {
        std::cout << "ConcatenateChannelWise::~ConcatenateChannelWise()" << '\n';
    }

    int Alloc(int noOperator, ...) {
        #ifdef __DEBUG__
        std::cout << "ConcatenateChannelWise::Alloc(Operator *, Operator *)" << '\n';
        #endif  // __DEBUG__

        m_noOperator = noOperator;

        va_list ap;
        va_start(ap, noOperator);

        Operator<DTYPE> *temp = va_arg(ap, Operator<DTYPE> *);

        int timesize         = temp->GetResult()->GetTimeSize();
        int batchsize        = temp->GetResult()->GetBatchSize();
        int totalchannelsize = temp->GetResult()->GetChannelSize();
        int rowsize          = temp->GetResult()->GetRowSize();
        int colsize          = temp->GetResult()->GetColSize();

        for (int i = 1; i < noOperator; i++) {
            temp = va_arg(ap, Operator<DTYPE> *);

            totalchannelsize += temp->GetResult()->GetChannelSize();
        }

        va_end(ap);

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, totalchannelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, totalchannelsize, rowsize, colsize));

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

        int totalchannelsize = 0;

        for (int opnum = 0; opnum < m_noOperator; opnum++) {
            input         = this->GetInput()[opnum]->GetResult();
            inputTenShape = input->GetShape();
            int channelsize = input->GetChannelSize();

            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            (*result)[Index5D(resultTenShape, ti, ba, totalchannelsize + ch, ro, co)]
                                = (*input)[Index5D(inputTenShape, ti, ba, ch, ro, co)];
                        }
                    }
                }
            }

            totalchannelsize += channelsize;
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

        int totalchannelsize = 0;

        for (int opnum = 0; opnum < m_noOperator; opnum++) {
            input_delta   = this->GetInput()[opnum]->GetDelta();
            inputTenShape = input_delta->GetShape();
            int channelsize = input_delta->GetChannelSize();

            for (int ba = 0; ba < batchsize; ba++) {
                for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < rowsize; ro++) {
                        for (int co = 0; co < colsize; co++) {
                            (*input_delta)[Index5D(inputTenShape, ti, ba, ch, ro, co)]
                                += (*this_delta)[Index5D(resultTenShape, ti, ba, totalchannelsize + ch, ro, co)];
                        }
                    }
                }
            }

            totalchannelsize += channelsize;
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

#endif  // CONCATENATECHANNELWISE_H_
