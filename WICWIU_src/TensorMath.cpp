#include "Tensor.hpp"

template class Tensor<float>;

#ifdef __CUDNN__
    template <typename DTYPE> void ArgmaxOnGPU(Tensor<DTYPE> *resultTensor, Tensor<DTYPE> *inputTensor, int dim);
#endif // __CUDNN__

template <typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Argmax(int dim) {
    if (dim > 4 || dim < -4) {
        std::cout << "Invalid Dim" << std::endl;
        exit(-1);
    }
    else if (dim < 0)
        dim = dim + 5;


    Shape *pShape = this->GetShape();
    int shape[5] = {(*pShape)[0], (*pShape)[1], (*pShape)[2], (*pShape)[3], (*pShape)[4]};
    shape[dim] /= (*pShape)[dim];


    Tensor<DTYPE> *resultTensor = new Tensor<DTYPE>(shape[0], shape[1], shape[2], shape[3], shape[4]);
    Shape *pReducedShape = resultTensor->GetShape();

    #ifdef __CUDNN__
        resultTensor->SetDeviceGPU(this->GetDeviceID(), TRUE);
        ArgmaxOnGPU<DTYPE>(resultTensor, this, dim);
        return resultTensor;
    #else // __CUDNN__

        int reducedSize = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];


        float maxTemp[reducedSize];

        for(int d=0; d<(*pShape)[dim]; d++) {
            for(int ti=0; ti<shape[0]; ti++) {
                for(int ba=0; ba<shape[1]; ba++) {
                    for(int ch=0; ch<shape[2]; ch++) {
                        for(int ro=0; ro<shape[3]; ro++) {
                            for(int co=0; co<shape[4]; co++) {
                                int temp[5] = {ti, ba, ch, ro, co};
                                temp[dim] = d;

                                int reducedIdx = Index5D(pReducedShape, ti, ba, ch, ro, co);
                                int thisIdx = Index5D(pShape, temp[0], temp[1], temp[2], temp[3], temp[4]);

                                if(d==0) {
                                    maxTemp[reducedIdx] = (*this)[thisIdx];
                                    (*resultTensor)[reducedIdx] = (float)d;
                                }
                                else if(maxTemp[reducedIdx] < (*this)[thisIdx]) {
                                    maxTemp[reducedIdx] = (*this)[thisIdx];
                                    (*resultTensor)[reducedIdx] = (float)d;
                                }
                            }
                        }
                    }
                }
            }
        }
    #endif

    return resultTensor;
}

template <typename DTYPE> void Tensor<DTYPE>::TriangleUpper(int standardDiagonal) {
    if (this->GetTimeSize() > 1 || this->GetBatchSize() > 1 || this->GetChannelSize() > 1) {
        std::cout << "Error: TriangleUpper must be operated on 2D Matrix But Got Tensor" << '\n';
        return;
    }
    int rowsize = this->GetRowSize();
    int colsize = this->GetColSize();
    for (int ro = 0; ro < rowsize; ro++) {
        for (int co = 0; co < ro + standardDiagonal; co++) {
            if (co >= colsize)
                break;
            (*this)[Index5D(this->GetShape(), 0, 0, 0, ro, co)] = 0;
        }
    }
}
template <typename DTYPE> void Tensor<DTYPE>::TriangleLower(int standardDiagonal) {
    if (this->GetTimeSize() > 1 || this->GetBatchSize() > 1 || this->GetChannelSize() > 1) {
        std::cout << "Error: TriangleLower must be operated on 2D Matrix But Got Tensor" << '\n';
        return;
    }
    int rowsize = this->GetRowSize();
    int colsize = this->GetColSize();
    for (int ro = 0; ro < rowsize; ro++) {
        for (int co = ro + standardDiagonal + 1; co < colsize; co++) {
            (*this)[Index5D(this->GetShape(), 0, 0, 0, ro, co)] = 0;
        }
    }
}

template <typename DTYPE> void Tensor<DTYPE>::TriangleUpper(int diagonal, DTYPE mask) {
    Shape *pshape = this->GetShape();

    int timesize    = (*pshape)[0];
    int batchsize   = (*pshape)[1];
    int channelsize = (*pshape)[2];
    int rowsize     = (*pshape)[3];
    int colsize     = (*pshape)[4];

    for (int ti = 0; ti < timesize; ti++) {
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < ro + diagonal; co++) {
                        if (co >= colsize)
                            break;
                        (*this)[Index5D(pshape, ti, ba, ch, ro, co)] = mask;
                    }
                }
            }
        }
    }
}

template <typename DTYPE> void Tensor<DTYPE>::TriangleLower(int diagonal, DTYPE mask) {

    Shape *pshape = this->GetShape();

    int timesize    = (*pshape)[0];
    int batchsize   = (*pshape)[1];
    int channelsize = (*pshape)[2];
    int rowsize     = (*pshape)[3];
    int colsize     = (*pshape)[4];

    for (int ti = 0; ti < timesize; ti++) {
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = ro + diagonal + 1; co < colsize; co++) {
                        (*this)[Index5D(pshape, ti, ba, ch, ro, co)] = mask;
                    }
                }
            }
        }
    }
}

template <typename DTYPE> void Tensor<DTYPE>::Clip(float min, float max) {
#ifdef __DEBUG__
    std::cout << "Tensor<DTYPE>::Clip()" << '\n';
#endif // __DEBUG__

    int timesize    = this->GetTimeSize();
    int batchsize   = this->GetBatchSize();
    int channelsize = this->GetChannelSize();
    int rowsize     = this->GetRowSize();
    int colsize     = this->GetColSize();

    Shape *resultTenShape = this->GetShape();
    int    index          = 0;

    int ti = 0;
    for (int ba = 0; ba < batchsize; ba++) {
        for (int ch = 0; ch < channelsize; ch++) {
            for (int ro = 0; ro < rowsize; ro++) {
                for (int co = 0; co < colsize; co++) {
                    index = (((ti * (*resultTenShape)[1] + ba) * (*resultTenShape)[2] + ch) * (*resultTenShape)[3] + ro) * (*resultTenShape)[4] + co;
                    if ((*m_aLongArray)[index] < min)
                        (*m_aLongArray)[index] = min;
                    else if ((*m_aLongArray)[index] > max)
                        (*m_aLongArray)[index] = max;
                }
            }
        }
    }

}

template <typename DTYPE> void Tensor<DTYPE>::MultiplyScalar(unsigned int pTime, float pScalar) {
    int batchsize   = this->GetBatchSize();
    int channelsize = this->GetChannelSize();
    int rowsize     = this->GetRowSize();
    int colsize     = this->GetColSize();
    int capacity    = batchsize * channelsize * rowsize * colsize;

    int ti = pTime;

    for (int index = ti * capacity; index < (ti + 1) * capacity; index++) {
        (*m_aLongArray)[index] = (*m_aLongArray)[index] * pScalar;
    }
}

template <typename DTYPE> void Tensor<DTYPE>::Power(unsigned int pTime, float pScalar) {
    int batchsize   = this->GetBatchSize();
    int channelsize = this->GetChannelSize();
    int rowsize     = this->GetRowSize();
    int colsize     = this->GetColSize();
    int capacity    = batchsize * channelsize * rowsize * colsize;

    int ti = pTime;

    for (int index = ti * capacity; index < (ti + 1) * capacity; index++) {
        (*m_aLongArray)[index] = (DTYPE)pow((*m_aLongArray)[index], pScalar);
    }
}

template <typename DTYPE> void Tensor<DTYPE>::GetSumTensor(Tensor<DTYPE> *pInputTensor, Tensor<DTYPE> *pOutputTensor, int dim) {
    if (dim > 4 || dim < -5) {
        std::cout << "Out of Range" << '\n';
        return;
    }
    if (dim < 0) {
        dim = dim + 5;
    }

    Shape *pInputShape  = pInputTensor->GetShape();
    Shape *pOutputShape = pOutputTensor->GetShape();

    int *aSumTensorDim = new int[5];

    for (int i = 0; i < 5; i++) {
        aSumTensorDim[i] = pOutputShape->GetDim(i);
    }
    aSumTensorDim[dim] = 1;

    int timeSize    = pInputShape->GetDim(0);
    int batchSize   = pInputShape->GetDim(1);
    int channelSize = pInputShape->GetDim(2);
    int rowSize     = pInputShape->GetDim(3);
    int columnSize  = pInputShape->GetDim(4);

    if (dim == 0) {
        for (int ba = 0; ba < batchSize; ba++) {
            for (int ch = 0; ch < channelSize; ch++) {
                for (int ro = 0; ro < rowSize; ro++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE sum = 0;
                        for (int ti = 0; ti < timeSize; ti++) {
                            sum += (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, 0, ba, ch, ro, co)] = sum;
                    }
                }
            }
        }
    }
    else if (dim == 1) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ch = 0; ch < channelSize; ch++) {
                for (int ro = 0; ro < rowSize; ro++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE sum = 0;
                        for (int ba = 0; ba < batchSize; ba++) {
                            sum += (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, ti, 0, ch, ro, co)] = sum;
                    }
                }
            }
        }
    }
    else if (dim == 2) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ba = 0; ba < batchSize; ba++) {
                for (int ro = 0; ro < rowSize; ro++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE sum = 0;
                        for (int ch = 0; ch < channelSize; ch++) {
                            sum += (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, ti, ba, 0, ro, co)] = sum;
                    }
                }
            }
        }
    }
    else if (dim == 3) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ba = 0; ba < batchSize; ba++) {
                for (int ch = 0; ch < channelSize; ch++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE sum = 0;
                        for (int ro = 0; ro < rowSize; ro++) {
                            sum += (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, ti, ba, ch, 0, co)] = sum;
                    }
                }
            }
        }
    }
    else if (dim == 4) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ba = 0; ba < batchSize; ba++) {
                for (int ch = 0; ch < channelSize; ch++) {
                    for (int ro = 0; ro < rowSize; ro++) {
                        DTYPE sum = 0;
                        for (int co = 0; co < columnSize; co++) {
                            sum += (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, ti, ba, ch, ro, 0)] = sum;
                    }
                }
            }
        }
    }
    delete[] aSumTensorDim;
}

template <typename DTYPE> void Tensor<DTYPE>::GetSumTensorOverAxes(Tensor<DTYPE> *pInputTensor, Tensor<DTYPE> *pOutputTensor, int numArgs, ...) {
    va_list ap;
    va_start(ap, numArgs);

    int *dims     = new int[numArgs];
    int  shape[5] = { pInputTensor->GetTimeSize(), pInputTensor->GetBatchSize(), pInputTensor->GetChannelSize(), pInputTensor->GetRowSize(),
                     pInputTensor->GetColSize() };

    for (int i = 0; i < numArgs; i++) {
        dims[i] = va_arg(ap, int);
        if (dims[i] < 0)
            dims[i] += 5;
    }
    std::sort(dims, dims + numArgs);
    std::vector<Tensor<DTYPE> *> tensors;
    for (int i = 0; i < numArgs; i++) {
        shape[dims[i]] = 1;
        if (i != numArgs - 1)
            tensors.push_back(new Tensor<DTYPE>(shape[0], shape[1], shape[2], shape[3], shape[4]));
    }
    if (numArgs == 1)
        GetSumTensor(pInputTensor, pOutputTensor, dims[0]);
    else {
        GetSumTensor(pInputTensor, tensors[0], dims[0]);
        for (int idx = 1; idx < numArgs - 1; idx++) {
            GetSumTensor(tensors[idx - 1], tensors[idx], dims[idx]);
        }
        GetSumTensor(tensors[numArgs - 2], pOutputTensor, dims[numArgs - 1]);
    }

    va_end(ap);
    for (int i = 0; i < tensors.size(); i++)
        delete tensors[i];
    tensors.clear();
    delete[] dims;
}

template <typename DTYPE> void Tensor<DTYPE>::GetSquaredSumTensorOverAxes(Tensor<DTYPE> *pInputTensor, Tensor<DTYPE> *pOutputTensor, int numArgs, ...) {
    va_list ap;
    va_start(ap, numArgs);

    int *dims     = new int[numArgs];
    int  shape[5] = { pInputTensor->GetTimeSize(), pInputTensor->GetBatchSize(), pInputTensor->GetChannelSize(), pInputTensor->GetRowSize(),
                     pInputTensor->GetColSize() };

    for (int i = 0; i < numArgs; i++) {
        dims[i] = va_arg(ap, int);
        if (dims[i] < 0)
            dims[i] += 5;
    }
    std::sort(dims, dims + numArgs);
    std::vector<Tensor<DTYPE> *> tensors;
    tensors.push_back(new Tensor<DTYPE>(shape[0], shape[1], shape[2], shape[3], shape[4]));
    Tensor<DTYPE> *squaredInputTensor = tensors.at(0);
    Shape         *pShape             = pInputTensor->GetShape();
    for (int ti = 0; ti < shape[0]; ti++) {
        for (int ba = 0; ba < shape[1]; ba++) {
            for (int ch = 0; ch < shape[2]; ch++) {
                for (int ro = 0; ro < shape[3]; ro++) {
                    for (int co = 0; co < shape[4]; co++) {
                        int index                    = Index5D(pShape, ti, ba, ch, ro, co);
                        (*squaredInputTensor)[index] = std::pow((*pInputTensor)[index], 2);
                    }
                }
            }
        }
    }
    for (int i = 0; i < numArgs; i++) {
        shape[dims[i]] = 1;
        if (i != numArgs - 1)
            tensors.push_back(new Tensor<DTYPE>(shape[0], shape[1], shape[2], shape[3], shape[4]));
    }
    if (numArgs == 1)
        GetSumTensor(tensors[0], pOutputTensor, dims[0]);
    else {
        for (int idx = 1; idx < numArgs; idx++) {
            GetSumTensor(tensors[idx - 1], tensors[idx], dims[idx - 1]);
        }
        GetSumTensor(tensors[numArgs - 1], pOutputTensor, dims[numArgs - 1]);
    }

    va_end(ap);
    for (int i = 0; i < tensors.size(); i++)
        delete tensors[i];
    tensors.clear();
    delete[] dims;
}

template <typename DTYPE> void Tensor<DTYPE>::GetVarTensor(Tensor<DTYPE> *pInputTensor, Tensor<DTYPE> *pOutputTensor, int dim, int unbiased) {

    if (dim > 4 || dim < -5) {
        std::cout << "Out of Range" << '\n';
        return;
    }
    if (dim < 0) {
        dim = dim + 5;
    }

    Shape *pInputShape  = pInputTensor->GetShape();
    Shape *pOutputShape = pOutputTensor->GetShape();

    int *aVarTensorDim = new int[5];

    for (int i = 0; i < 5; i++) {
        aVarTensorDim[i] = pOutputShape->GetDim(i);
    }
    aVarTensorDim[dim] = 1;

    int timeSize    = pInputShape->GetDim(0);
    int batchSize   = pInputShape->GetDim(1);
    int channelSize = pInputShape->GetDim(2);
    int rowSize     = pInputShape->GetDim(3);
    int columnSize  = pInputShape->GetDim(4);

    int numberOfData     = pInputShape->GetDim(dim);
    int correctionNumber = numberOfData;
    if (unbiased && numberOfData > 1)
        correctionNumber -= 1;

    if (dim == 0) {
        for (int ba = 0; ba < batchSize; ba++) {
            for (int ch = 0; ch < channelSize; ch++) {
                for (int ro = 0; ro < rowSize; ro++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE sum        = 0;
                        DTYPE squaredSum = 0;
                        for (int ti = 0; ti < timeSize; ti++) {
                            DTYPE value = (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                            sum += value;
                            squaredSum += std::pow(value, (DTYPE)2);
                        }
                        DTYPE variance = (squaredSum - std::pow(sum, 2) / (DTYPE)numberOfData) / (correctionNumber);
                        (*pOutputTensor)[Index5D(pOutputShape, 0, ba, ch, ro, co)] = variance;
                    }
                }
            }
        }
    }
    else if (dim == 1) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ch = 0; ch < channelSize; ch++) {
                for (int ro = 0; ro < rowSize; ro++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE sum        = 0;
                        DTYPE squaredSum = 0;
                        for (int ba = 0; ba < batchSize; ba++) {
                            DTYPE value = (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                            sum += value;
                            squaredSum += std::pow(value, (DTYPE)2);
                        }
                        DTYPE variance = (squaredSum - std::pow(sum, 2) / (DTYPE)numberOfData) / (correctionNumber);
                        (*pOutputTensor)[Index5D(pOutputShape, ti, 0, ch, ro, co)] = variance;
                    }
                }
            }
        }
    }
    else if (dim == 2) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ba = 0; ba < batchSize; ba++) {
                for (int ro = 0; ro < rowSize; ro++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE sum        = 0;
                        DTYPE squaredSum = 0;
                        for (int ch = 0; ch < channelSize; ch++) {
                            DTYPE value = (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                            sum += value;
                            squaredSum += std::pow(value, (DTYPE)2);
                        }
                        DTYPE variance = (squaredSum - std::pow(sum, 2) / (DTYPE)numberOfData) / (correctionNumber);
                        (*pOutputTensor)[Index5D(pOutputShape, ti, ba, 0, ro, co)] = variance;
                    }
                }
            }
        }
    }
    else if (dim == 3) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ba = 0; ba < batchSize; ba++) {
                for (int ch = 0; ch < channelSize; ch++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE sum        = 0;
                        DTYPE squaredSum = 0;
                        for (int ro = 0; ro < rowSize; ro++) {
                            DTYPE value = (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                            sum += value;
                            squaredSum += std::pow(value, (DTYPE)2);
                        }
                        DTYPE variance = (squaredSum - std::pow(sum, 2) / (DTYPE)numberOfData) / (correctionNumber);
                        (*pOutputTensor)[Index5D(pOutputShape, ti, ba, ch, 0, co)] = variance;
                    }
                }
            }
        }
    }
    else if (dim == 4) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ba = 0; ba < batchSize; ba++) {
                for (int ch = 0; ch < channelSize; ch++) {
                    for (int ro = 0; ro < rowSize; ro++) {
                        DTYPE sum        = 0;
                        DTYPE squaredSum = 0;
                        for (int co = 0; co < columnSize; co++) {
                            DTYPE value = (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                            sum += value;
                            squaredSum += std::pow(value, (DTYPE)2);
                        }
                        DTYPE variance = (squaredSum - std::pow(sum, 2) / (DTYPE)numberOfData) / (correctionNumber);
                        (*pOutputTensor)[Index5D(pOutputShape, ti, ba, ch, ro, 0)] = variance;
                    }
                }
            }
        }
    }
    delete[] aVarTensorDim;
}

template <typename DTYPE> void Tensor<DTYPE>::GetMeanTensor(Tensor<DTYPE> *pInputTensor, Tensor<DTYPE> *pOutputTensor, int dim) {

    if (dim > 4 || dim < -5) {
        std::cout << "Out of Range" << '\n';
        return;
    }
    if (dim < 0) {
        dim = dim + 5;
    }

    Shape *pInputShape  = pInputTensor->GetShape();
    Shape *pOutputShape = pOutputTensor->GetShape();

    int *aMeanTensorDim = new int[5];

    for (int i = 0; i < 5; i++) {
        aMeanTensorDim[i] = pOutputShape->GetDim(i);
    }
    aMeanTensorDim[dim] = 1;

    int timeSize    = pInputShape->GetDim(0);
    int batchSize   = pInputShape->GetDim(1);
    int channelSize = pInputShape->GetDim(2);
    int rowSize     = pInputShape->GetDim(3);
    int columnSize  = pInputShape->GetDim(4);

    int numberOfData = pInputShape->GetDim(dim);

    if (dim == 0) {
        for (int ba = 0; ba < batchSize; ba++) {
            for (int ch = 0; ch < channelSize; ch++) {
                for (int ro = 0; ro < rowSize; ro++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE sum = 0;
                        for (int ti = 0; ti < timeSize; ti++) {
                            sum += (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, 0, ba, ch, ro, co)] = sum / (DTYPE)numberOfData;
                    }
                }
            }
        }
    }
    else if (dim == 1) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ch = 0; ch < channelSize; ch++) {
                for (int ro = 0; ro < rowSize; ro++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE sum = 0;
                        for (int ba = 0; ba < batchSize; ba++) {
                            sum += (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, ti, 0, ch, ro, co)] = sum / (DTYPE)numberOfData;
                    }
                }
            }
        }
    }
    else if (dim == 2) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ba = 0; ba < batchSize; ba++) {
                for (int ro = 0; ro < rowSize; ro++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE sum = 0;
                        for (int ch = 0; ch < channelSize; ch++) {
                            sum += (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, ti, ba, 0, ro, co)] = sum / (DTYPE)numberOfData;
                    }
                }
            }
        }
    }
    else if (dim == 3) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ba = 0; ba < batchSize; ba++) {
                for (int ch = 0; ch < channelSize; ch++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE sum = 0;
                        for (int ro = 0; ro < rowSize; ro++) {
                            sum += (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, ti, ba, ch, 0, co)] = sum / (DTYPE)numberOfData;
                    }
                }
            }
        }
    }
    else if (dim == 4) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ba = 0; ba < batchSize; ba++) {
                for (int ch = 0; ch < channelSize; ch++) {
                    for (int ro = 0; ro < rowSize; ro++) {
                        DTYPE sum = 0;
                        for (int co = 0; co < columnSize; co++) {
                            sum += (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, ti, ba, ch, ro, 0)] = sum / (DTYPE)numberOfData;
                    }
                }
            }
        }
    }
    delete[] aMeanTensorDim;
}
template <typename DTYPE> void Tensor<DTYPE>::GetMaxTensor(Tensor<DTYPE> *pInputTensor, Tensor<DTYPE> *pOutputTensor, int dim) {
    if (dim > 4 || dim < -5) {
        std::cout << "Out of Range" << '\n';
        return;
    }
    if (dim < 0) {
        dim = dim + 5;
    }

    Shape *pInputShape  = pInputTensor->GetShape();
    Shape *pOutputShape = pOutputTensor->GetShape();

    int *aMaxTensorDim = new int[5];

    for (int i = 0; i < 5; i++) {
        aMaxTensorDim[i] = pOutputShape->GetDim(i);
    }
    aMaxTensorDim[dim] = 1;

    int timeSize    = pInputShape->GetDim(0);
    int batchSize   = pInputShape->GetDim(1);
    int channelSize = pInputShape->GetDim(2);
    int rowSize     = pInputShape->GetDim(3);
    int columnSize  = pInputShape->GetDim(4);

    if (dim == 0) {
        for (int ba = 0; ba < batchSize; ba++) {
            for (int ch = 0; ch < channelSize; ch++) {
                for (int ro = 0; ro < rowSize; ro++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE max = 0;
                        for (int ti = 0; ti < timeSize; ti++) {
                            DTYPE value = (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                            if (max < value)
                                max = value;
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, 0, ba, ch, ro, co)] = max;
                    }
                }
            }
        }
    }
    else if (dim == 1) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ch = 0; ch < channelSize; ch++) {
                for (int ro = 0; ro < rowSize; ro++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE max = 0;
                        for (int ba = 0; ba < batchSize; ba++) {
                            DTYPE value = (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                            if (max < value)
                                max = value;
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, ti, 0, ch, ro, co)] = max;
                    }
                }
            }
        }
    }
    else if (dim == 2) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ba = 0; ba < batchSize; ba++) {
                for (int ro = 0; ro < rowSize; ro++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE max = 0;
                        for (int ch = 0; ch < channelSize; ch++) {
                            DTYPE value = (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                            if (max < value)
                                max = value;
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, ti, ba, 0, ro, co)] = max;
                    }
                }
            }
        }
    }
    else if (dim == 3) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ba = 0; ba < batchSize; ba++) {
                for (int ch = 0; ch < channelSize; ch++) {
                    for (int co = 0; co < columnSize; co++) {
                        DTYPE max = 0;
                        for (int ro = 0; ro < rowSize; ro++) {
                            DTYPE value = (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                            if (max < value)
                                max = value;
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, ti, ba, ch, 0, co)] = max;
                    }
                }
            }
        }
    }
    else if (dim == 4) {
        for (int ti = 0; ti < timeSize; ti++) {
            for (int ba = 0; ba < batchSize; ba++) {
                for (int ch = 0; ch < channelSize; ch++) {
                    for (int ro = 0; ro < rowSize; ro++) {
                        DTYPE max = 0;
                        for (int co = 0; co < columnSize; co++) {
                            DTYPE value = (*pInputTensor)[Index5D(pInputShape, ti, ba, ch, ro, co)];
                            if (max < value)
                                max = value;
                        }
                        (*pOutputTensor)[Index5D(pOutputShape, ti, ba, ch, ro, 0)] = max;
                    }
                }
            }
        }
    }
    delete[] aMaxTensorDim;
}

template <typename DTYPE>
void Tensor<DTYPE>::GetVarTensorOverAxes(Tensor<DTYPE> *pInputTensor, Tensor<DTYPE> *pOutputTensor, int unbiased, int numArgs, ...) {
    va_list ap;
    va_start(ap, numArgs);

    int *dims = new int[numArgs];

    for (int i = 0; i < numArgs; i++) {
        dims[i] = va_arg(ap, int);
        if (dims[i] < 0)
            dims[i] += 5;
    }
    std::sort(dims, dims + numArgs);

    Shape *        resultShape = pOutputTensor->GetShape();
    Tensor<DTYPE> *sum = new Tensor<DTYPE>(resultShape->GetDim(0), resultShape->GetDim(1), resultShape->GetDim(2), resultShape->GetDim(3),
                                           resultShape->GetDim(4));
    Tensor<DTYPE> *squaredSum = new Tensor<DTYPE>(resultShape->GetDim(0), resultShape->GetDim(1), resultShape->GetDim(2),
                                                  resultShape->GetDim(3), resultShape->GetDim(4));
    if (numArgs == 1) {
        Tensor<DTYPE>::GetSumTensorOverAxes(pInputTensor, sum, numArgs, dims[0]);
        Tensor<DTYPE>::GetSquaredSumTensorOverAxes(pInputTensor, squaredSum, numArgs, dims[0]);
    }
    else if (numArgs == 2) {
        Tensor<DTYPE>::GetSumTensorOverAxes(pInputTensor, sum, numArgs, dims[0], dims[1]);
        Tensor<DTYPE>::GetSquaredSumTensorOverAxes(pInputTensor, squaredSum, numArgs, dims[0], dims[1]);
    }
    else if (numArgs == 3) {
        Tensor<DTYPE>::GetSumTensorOverAxes(pInputTensor, sum, numArgs, dims[0], dims[1], dims[2]);
        Tensor<DTYPE>::GetSquaredSumTensorOverAxes(pInputTensor, squaredSum, numArgs, dims[0], dims[1], dims[2]);
    }
    else if (numArgs == 4) {
        Tensor<DTYPE>::GetSumTensorOverAxes(pInputTensor, sum, numArgs, dims[0], dims[1], dims[2], dims[3]);
        Tensor<DTYPE>::GetSquaredSumTensorOverAxes(pInputTensor, squaredSum, numArgs, dims[0], dims[1], dims[2], dims[3]);
    }
    else if (numArgs == 5) {
        Tensor<DTYPE>::GetSumTensorOverAxes(pInputTensor, sum, numArgs, dims[0], dims[1], dims[2], dims[3], dims[4]);
        Tensor<DTYPE>::GetSquaredSumTensorOverAxes(pInputTensor, squaredSum, numArgs, dims[0], dims[1], dims[2], dims[3], dims[4]);
    }

    int shape[5] = { pInputTensor->GetTimeSize(), pInputTensor->GetBatchSize(), pInputTensor->GetChannelSize(), pInputTensor->GetRowSize(),
                     pInputTensor->GetColSize() };
    float num    = 1;
    for (int i = 0; i < numArgs; i++) {
        num *= shape[dims[i]];
        shape[dims[i]] = 1;
    }
    float correctionNum = num;
    if (unbiased && num > 1)
        correctionNum--;
    if (numArgs == 1)
        GetVarTensor(pInputTensor, pOutputTensor, dims[0], unbiased);
    else {
        for (int ti = 0; ti < shape[0]; ti++) {
            for (int ba = 0; ba < shape[1]; ba++) {
                for (int ch = 0; ch < shape[2]; ch++) {
                    for (int ro = 0; ro < shape[3]; ro++) {
                        for (int co = 0; co < shape[4]; co++) {
                            int index               = Index5D(resultShape, ti, ba, ch, ro, co);
                            (*pOutputTensor)[index] = ((*squaredSum)[index] - std::pow((*sum)[index], 2) / num) / (correctionNum);
                        }
                    }
                }
            }
        }
    }

    va_end(ap);
    delete sum;
    delete squaredSum;
    delete[] dims;
}

template <typename DTYPE> void Tensor<DTYPE>::GetMeanTensorOverAxes(Tensor<DTYPE> *pInputTensor, Tensor<DTYPE> *pOutputTensor, int numArgs, ...) {
    va_list ap;
    va_start(ap, numArgs);

    int *dims     = new int[numArgs];
    int  shape[5] = { pInputTensor->GetTimeSize(), pInputTensor->GetBatchSize(), pInputTensor->GetChannelSize(), pInputTensor->GetRowSize(),
                     pInputTensor->GetColSize() };
    std::vector<Tensor<DTYPE> *> tensors;
    for (int i = 0; i < numArgs; i++) {
        dims[i] = va_arg(ap, int);
        if (dims[i] < 0)
            dims[i] += 5;
    }
    std::sort(dims, dims + numArgs);
    for (int i = 0; i < numArgs; i++) {
        shape[dims[i]] = 1;
        if (i != numArgs - 1)
            tensors.push_back(new Tensor<DTYPE>(shape[0], shape[1], shape[2], shape[3], shape[4]));
    }
    if (numArgs == 1)
        GetMeanTensor(pInputTensor, pOutputTensor, dims[0]);
    else {
        GetMeanTensor(pInputTensor, tensors[0], dims[0]);
        for (int idx = 1; idx < numArgs - 1; idx++) {
            GetMeanTensor(tensors[idx - 1], tensors[idx], dims[idx]);
        }
        GetMeanTensor(tensors[numArgs - 2], pOutputTensor, dims[numArgs - 1]);
    }

    va_end(ap);
    for (int i = 0; i < tensors.size(); i++)
        delete tensors[i];
    tensors.clear();
    delete[] dims;
}

template <typename DTYPE> Tensor<DTYPE> &Tensor<DTYPE>::operator+(Tensor<DTYPE> &rTensor) {
    Shape *pShape = rTensor.GetShape();

    int timesize    = GetTimeSize();
    int batchsize   = GetBatchSize();
    int channelsize = GetChannelSize();
    int rowsize     = GetRowSize();
    int colsize     = GetColSize();

    int timesize2    = rTensor.GetTimeSize();
    int batchsize2   = rTensor.GetBatchSize();
    int channelsize2 = rTensor.GetChannelSize();
    int rowsize2     = rTensor.GetRowSize();
    int colsize2     = rTensor.GetColSize();

    try {
        std::string dim;
        if (timesize != timesize2)
            dim += " Time";
        if (batchsize != batchsize2)
            dim += " Batch";
        if (channelsize != channelsize2)
            dim += " Channel";
        if (rowsize != rowsize2)
            dim += " Row";
        if (colsize != colsize2)
            dim += " Col";

        if (!dim.empty())
            throw(dim);
    }
    catch (std::string dim) {
        std::cout << dim << " Size Unmatch.." << std::endl;
        exit(-1);
    }

    for (int ti = 0; ti < timesize; ti++) {
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        (*this)[Index5D(this->GetShape(), ti, ba, ch, ro, co)] += rTensor[Index5D(this->GetShape(), ti, ba, ch, ro, co)];
                    }
                }
            }
        }
    }

    return *this;
}

template <typename DTYPE> Tensor<DTYPE> &Tensor<DTYPE>::operator-(Tensor<DTYPE> &rTensor) {
    Shape *pShape = rTensor.GetShape();

    int timesize    = GetTimeSize();
    int batchsize   = GetBatchSize();
    int channelsize = GetChannelSize();
    int rowsize     = GetRowSize();
    int colsize     = GetColSize();

    int timesize2    = rTensor.GetTimeSize();
    int batchsize2   = rTensor.GetBatchSize();
    int channelsize2 = rTensor.GetChannelSize();
    int rowsize2     = rTensor.GetRowSize();
    int colsize2     = rTensor.GetColSize();

    try {
        std::string dim;
        if (timesize != timesize2)
            dim += " Time";
        if (batchsize != batchsize2)
            dim += " Batch";
        if (channelsize != channelsize2)
            dim += " Channel";
        if (rowsize != rowsize2)
            dim += " Row";
        if (colsize != colsize2)
            dim += " Col";

        if (!dim.empty())
            throw(dim);
    }
    catch (std::string dim) {
        std::cout << dim << " Size Unmatch.." << std::endl;
        exit(-1);
    }

    for (int ti = 0; ti < timesize; ti++) {
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        (*this)[Index5D(this->GetShape(), ti, ba, ch, ro, co)] -= rTensor[Index5D(this->GetShape(), ti, ba, ch, ro, co)];
                    }
                }
            }
        }
    }

    return *this;
}

template <typename DTYPE> Tensor<DTYPE> &Tensor<DTYPE>::operator*(Tensor<DTYPE> &rTensor) {
    Shape *pShape = rTensor.GetShape();

    int timesize    = GetTimeSize();
    int batchsize   = GetBatchSize();
    int channelsize = GetChannelSize();
    int rowsize     = GetRowSize();
    int colsize     = GetColSize();

    int timesize2    = rTensor.GetTimeSize();
    int batchsize2   = rTensor.GetBatchSize();
    int channelsize2 = rTensor.GetChannelSize();
    int rowsize2     = rTensor.GetRowSize();
    int colsize2     = rTensor.GetColSize();

    try {
        std::string dim;
        if (timesize != timesize2)
            dim += " Time";
        if (batchsize != batchsize2)
            dim += " Batch";
        if (channelsize != channelsize2)
            dim += " Channel";
        if (rowsize != rowsize2)
            dim += " Row";
        if (colsize != colsize2)
            dim += " Col";

        if (!dim.empty())
            throw(dim);
    }
    catch (std::string dim) {
        std::cout << dim << " Size Unmatch.." << std::endl;
        exit(-1);
    }

    for (int ti = 0; ti < timesize; ti++) {
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        (*this)[Index5D(this->GetShape(), ti, ba, ch, ro, co)] *= rTensor[Index5D(this->GetShape(), ti, ba, ch, ro, co)];
                    }
                }
            }
        }
    }

    return *this;
}

template <typename DTYPE> Tensor<DTYPE> &Tensor<DTYPE>::operator/(Tensor<DTYPE> &rTensor) {
    Shape *pShape = rTensor.GetShape();

    int timesize    = GetTimeSize();
    int batchsize   = GetBatchSize();
    int channelsize = GetChannelSize();
    int rowsize     = GetRowSize();
    int colsize     = GetColSize();

    int timesize2    = rTensor.GetTimeSize();
    int batchsize2   = rTensor.GetBatchSize();
    int channelsize2 = rTensor.GetChannelSize();
    int rowsize2     = rTensor.GetRowSize();
    int colsize2     = rTensor.GetColSize();

    try {
        std::string dim;
        if (timesize != timesize2)
            dim += " Time";
        if (batchsize != batchsize2)
            dim += " Batch";
        if (channelsize != channelsize2)
            dim += " Channel";
        if (rowsize != rowsize2)
            dim += " Row";
        if (colsize != colsize2)
            dim += " Col";

        if (!dim.empty())
            throw(dim);
    }
    catch (std::string dim) {
        std::cout << dim << " Size Unmatch.." << std::endl;
        exit(-1);
    }

    for (int ti = 0; ti < timesize; ti++) {
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        (*this)[Index5D(this->GetShape(), ti, ba, ch, ro, co)] /= rTensor[Index5D(this->GetShape(), ti, ba, ch, ro, co)];
                    }
                }
            }
        }
    }

    return *this;
}
