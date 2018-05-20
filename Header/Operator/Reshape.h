#ifndef RESHAPE_H_
#define RESHAPE_H_    value

#include "..//Operator.h"

template<typename DTYPE>
class Reshape : public Operator<DTYPE>{
private:
public:
    Reshape(Operator<DTYPE> *pInput, int pRowSize, int pColSize, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #if __DEBUG__
        std::cout << "Reshape::Reshape(Operator *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, 0, 0, 0, pRowSize, pColSize);
    }

    Reshape(Operator<DTYPE> *pInput, int pChannelSize, int pRowSize, int pColSize, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #if __DEBUG__
        std::cout << "Reshape::Reshape(Operator *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, 0, 0, pChannelSize, pRowSize, pColSize);
    }

    Reshape(Operator<DTYPE> *pInput, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #if __DEBUG__
        std::cout << "Reshape::Reshape(Operator *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, 0, pBatchSize, pChannelSize, pRowSize, pColSize);
    }

    Reshape(Operator<DTYPE> *pInput, int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #if __DEBUG__
        std::cout << "Reshape::Reshape(Operator *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);
    }

    ~Reshape() {
        #if __DEBUG__
        std::cout << "Reshape::~Reshape()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
        #if __DEBUG__
        std::cout << "Reshape::Alloc(Operator *, Operator *)" << '\n';
        #endif  // __DEBUG__

        Shape *pInputShape = pInput->GetResult()->GetShape();

        if (pTimeSize == 0) pTimeSize = (*pInputShape)[0];

        if (pBatchSize == 0) pBatchSize = (*pInputShape)[1];

        if (pChannelSize == 0) pChannelSize = (*pInputShape)[2];


        Tensor<DTYPE> *result = new Tensor<DTYPE>(pInput->GetResult());
        result->Reshape(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);

        this->SetResult(result);  // copy data

        this->SetDelta(new Tensor<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize));

        return TRUE;
    }

    void Delete() {}

    int  ForwardPropagate(int pTime = 0, int pThreadNum = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        int ti          = pTime;
        int numOfThread = this->GetNumOfThread();

        for (int ba = pThreadNum; ba < batchsize; ba += numOfThread) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                            = (*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)];
                    }
                }
            }
        }

        return TRUE;
    }

    int BackPropagate(int pTime = 0, int pThreadNum = 0) {
        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        int timesize    = this_delta->GetTimeSize();
        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize     = this_delta->GetRowSize();
        int colsize     = this_delta->GetColSize();

        Shape *deltaTenShape = this_delta->GetShape();

        int ti          = pTime;
        int numOfThread = this->GetNumOfThread();

        for (int ba = pThreadNum; ba < batchsize; ba += numOfThread) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        (*input_delta)[Index5D(deltaTenShape, ti, ba, ch, ro, co)]
                            += (*this_delta)[Index5D(deltaTenShape, ti, ba, ch, ro, co)];
                    }
                }
            }
        }

        return TRUE;
    }

#if __CUDNN__
    int ForwardPropagateOnGPU(int pTime) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        DTYPE *pDevInput  = input->GetDeviceData();
        DTYPE *pDevResult = result->GetDeviceData();

        cudnnTensorDescriptor_t pDesc = input->GetDescriptor();

        float alpha = 1.f;
        float beta  = 0.f;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &alpha, pDesc, pDevInput,
                                  &beta, pDesc, pDevResult));

        // this->ForwardPropagate(pTime);
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime) {
        Tensor<DTYPE> *this_delta  = this->GetDelta();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        DTYPE *pDevDelta  = this_delta->GetDeviceData();
        DTYPE *pDevInputDelta = input_delta->GetDeviceData();

        cudnnTensorDescriptor_t pDesc = this_delta->GetDescriptor();

        float alpha = 1.f;
        float beta  = 0.f;

        checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                  &alpha, pDesc, pDevDelta,
                                  &beta, pDesc, pDevInputDelta));

        // this->BackPropagate(pTime);

        return TRUE;
    }

#endif  // __CUDNN__
};

#endif  // RESHAPE_H_
