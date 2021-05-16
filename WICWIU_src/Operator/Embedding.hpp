#ifndef EMBEDDING_H_
#define EMBEDDING_H_    value

#include "../Operator.hpp"
#include <cstdio>

template<typename DTYPE> class Embedding : public Operator<DTYPE>{
private:

public:

    Embedding(Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pWeight, pInput, pName, pLoadflag) {
       #ifdef __DEBUG__
        std::cout << "Embedding::Embedding(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
       #endif  // __DEBUG__
        this->Alloc(pWeight, pInput);
    }


    virtual ~Embedding() {
        #ifdef __DEBUG__
        std::cout << "Embedding::~Embedding()" << '\n';
        #endif  // __DEBUG__
        Delete();
    }


    int Alloc(Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput) {
        #ifdef __DEBUG__
        std::cout << "Embedding::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__


        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        //int channelsize = pInput->GetResult()->GetChannelSize();
        //int rowsize     = pInput->GetResult()->GetRowSize();
        int rowsize     = pInput->GetResult()->GetColSize();
        int embeddingsize     = pWeight->GetResult()->GetColSize();       //embedding size

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, rowsize, embeddingsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, 1, rowsize, embeddingsize));

        return TRUE;
    }



    void Delete() {}

    int ForwardPropagate(int pTime = 0) {

        Tensor<DTYPE> *weight = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input  = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize     = result->GetBatchSize();
        int channelsize   = result->GetChannelSize();
        int numOfWord       = result->GetRowSize();
        int embeddingDim     = result->GetColSize();

        Shape *weightTenShape = weight->GetShape();
        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = result->GetShape();

        int ti = pTime;
        int wordIndex=0;

        for (int ba = 0; ba < batchsize; ba++) {
              for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < numOfWord; ro++) {
                        wordIndex = ((*input)[Index5D(inputTenShape, ti, ba, ch, 0, ro)]);
                        for (int co = 0; co < embeddingDim; co++) {
                            (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                                = ((*weight)[Index5D(weightTenShape, 0, 0, 0, wordIndex, co)]);
                        }
                    }
              }
        }
        return TRUE;
    }


    int BackPropagate(int pTime = 0) {

        Tensor<DTYPE> *input           = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *thisDelta      = this->GetDelta();
        Tensor<DTYPE> *weightGradient = this->GetInput()[0]->GetGradient();

        int batchsize   = thisDelta->GetBatchSize();
        int channelsize = thisDelta->GetChannelSize();
        int rowsize     = thisDelta->GetRowSize();
        int numOfWord     = thisDelta->GetColSize();

        Shape *weightTenShape = weightGradient->GetShape();
        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = thisDelta->GetShape();

        int ti = pTime;
        int wordIndex=0;

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    wordIndex = ((*input)[Index5D(inputTenShape, ti, ba, ch, 0, ro)]);
                    for (int co = 0; co < numOfWord; co++) {
                        (*weightGradient)[Index5D(weightTenShape, 0, 0, 0, wordIndex, co)]
                            += ((*thisDelta)[Index5D(resultTenShape, ti, ba, ch, ro, co)]);
                    }
                }
            }
        }

        return TRUE;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime);

    int BackPropagateOnGPU(int pTime);

#endif  // __CUDNN__

};


#endif  // Embedding_H_
