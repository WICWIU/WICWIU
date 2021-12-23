#ifndef __TENSOR_UTIL_H__
#define __TENSOR_UTIL_H__    value

#include "Tensor.hpp"

template<typename DTYPE> std::ostream & operator<<(std::ostream& pOS, Tensor<DTYPE> *pTensor) {
    int timesize    = pTensor->GetTimeSize();
    int batchsize   = pTensor->GetBatchSize();
    int channelsize = pTensor->GetChannelSize();
    int rowsize     = pTensor->GetRowSize();
    int colsize     = pTensor->GetColSize();

    Shape *shape = pTensor->GetShape();

    pOS << "[";
    for(int ti = 0 ; ti < timesize ; ti++) {
        pOS << "[";
        for(int ba = 0 ; ba < batchsize ; ba++) {
            pOS << "[";
            for(int ch = 0 ; ch < channelsize ; ch++) {
                pOS << "[";
                for(int ro = 0 ; ro < rowsize ; ro++) {
                    pOS << "[";
                    for(int co = 0 ; co <colsize ; co++) {
                        char n[20];
                        sprintf(n, "%-10.6f", (*pTensor)[Index5D(shape, ti, ba, ch, ro, co)]);
                        pOS << n;
                        if(co != colsize-1) pOS << ", ";
                        else pOS << "]";
                    }
                    if(ro != rowsize-1) pOS << ",\n    ";
                }
                if(ch != channelsize-1) pOS << "],\n\n   ";
                else pOS << "]";
            }
            if(ba != batchsize-1) pOS << "],\n\n\n  ";
            else pOS << "]";
        }
        if(ti != timesize-1) pOS << "],\n\n\n\n ";
        else pOS << "]";
    }
    pOS<< "]";

    return pOS;
}

#endif  // __TENSOR_UTIL_H__
