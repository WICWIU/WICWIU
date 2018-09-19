#ifndef __TENSOR_UTIL_H__
#define __TENSOR_UTIL_H__    value

#include "Tensor.h"

template<typename DTYPE> std::ostream & operator<<(std::ostream& pOS, Tensor<DTYPE> *pTensor) {
    int timesize    = pTensor->GetTimeSize();
    int batchsize   = pTensor->GetBatchSize();
    int channelsize = pTensor->GetChannelSize();
    int rowsize     = pTensor->GetRowSize();
    int colsize     = pTensor->GetColSize();

    Shape *shape = pTensor->GetShape();

    pOS.precision(4);

    pOS << "[ \n";

    for (int ti = 0; ti < timesize; ti++) {
        pOS << "[ \n";

        for (int ba = 0; ba < batchsize; ba++) {
            pOS << "[ \n";

            for (int ch = 0; ch < channelsize; ch++) {
                pOS << "[ ";

                for (int ro = 0; ro < rowsize; ro++) {
                    pOS << "\t[ ";

                    for (int co = 0; co < colsize; co++) {
                        if (co != colsize - 1) {
                            pOS << "\t\t" << (*pTensor)[Index5D(shape, ti, ba, ch, ro, co)] << ", ";
                        }
                        else {
                            pOS << "\t\t" << (*pTensor)[Index5D(shape, ti, ba, ch, ro, co)];
                        }
                    }

                    if (ro != rowsize - 1) {
                        pOS << " \t\t]\n";
                    }
                    else {
                        pOS << " \t\t]";
                    }
                }
                pOS << " ]\n";
            }
            pOS << "]\n";
        }
        pOS << "]\n";
    }
    pOS << "]\n";

    return pOS;
}

#endif  // __TENSOR_UTIL_H__
