#ifndef EMBEDDING_H_
#define EMBEDDING_H_    value

#include "../Operator.hpp"
#include <cstdio>

template<typename DTYPE> class Embedding : public Operator<DTYPE>{
private:

public:

    Embedding(Operator<DTYPE> *pWeight, Operator<DTYPE> *pInput, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pWeight, pInput, pName, pLoadflag) {
//        #ifdef __DEBUG__
        std::cout << "Embedding::Embedding(Operator<DTYPE> *, Operator<DTYPE> *, std::string)" << '\n';
//        #endif  // __DEBUG__
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

        std::cout<<this->GetResult()->GetShape()<<'\n';

        return TRUE;
    }



    void Delete() {}

    int ForwardPropagate(int pTime = 0) {

        // std::cout<<"Embedding forward "<<pTime<<'\n';

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

        // std::cout<<"input의 shape및 값"<<'\n';
        // std::cout << input->GetShape() << '\n';
        // std::cout<<input<<'\n';
        //
        // std::cout<<"numOfWord : "<<numOfWord<<'\n';
        // std::cout<<"embeddingDim : "<<embeddingDim<<'\n';

        for (int ba = 0; ba < batchsize; ba++) {
              for (int ch = 0; ch < channelsize; ch++) {
                    for (int ro = 0; ro < numOfWord; ro++) {
                        wordIndex = ((*input)[Index5D(inputTenShape, ti, ba, ch, 0, ro)]);                //1이아니라 0부터 시작해야지!!!
                        if(wordIndex==-1)
                            std::cout<<"오류---------------------------입력값에 -1 존재!!!"<<'\n';
                        // std::cout<<"wordIndex : "<<wordIndex<<'\n';
                        for (int co = 0; co < embeddingDim; co++) {
                            (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                                = ((*weight)[Index5D(weightTenShape, 0, 0, 0, wordIndex, co)]);                  //weight 접근할때는 ti, ba, ch 다 0이다!!!
                        }
                    }
              }
        }
        return TRUE;
    }


    int BackPropagate(int pTime = 0) {

        //std::cout<<'\n'<<"Embedding backward 호출"<<'\n';

        Tensor<DTYPE> *input           = this->GetInput()[1]->GetResult();
        Tensor<DTYPE> *this_delta      = this->GetDelta();
        Tensor<DTYPE> *weight_gradient = this->GetInput()[0]->GetGradient();
        // Tensor<DTYPE> *input_delta     = this->GetInput()[1]->GetDelta();

        int batchsize   = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize     = this_delta->GetRowSize();
        int numOfWord     = this_delta->GetColSize();

        Shape *weightTenShape = weight_gradient->GetShape();
        Shape *inputTenShape  = input->GetShape();
        Shape *resultTenShape = this_delta->GetShape();

        int ti = pTime;
        int wordIndex=0;

       //std::cout<<'\n'<<"Embedding Operator 위에서 주는 gradient 값!"<<'\n';
       //std::cout<<resultTenShape<<'\n';
       //std::cout<<this_delta<<'\n';

        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    wordIndex = ((*input)[Index5D(inputTenShape, ti, ba, ch, 0, ro)]);
                    for (int co = 0; co < numOfWord; co++) {
                        (*weight_gradient)[Index5D(weightTenShape, 0, 0, 0, wordIndex, co)]
                            += ((*this_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)]);
                        //input으로 가는 delta 추가해줘야함!!!
                        // std::cout<<((*this_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)])<<" ";
                    }
                }
            }
        }

       //std::cout<<'\n'<<"-----------embedding backpropagate-------"<<'\n';
       //std::cout<<this->GetInput()[1]->GetGradient()<<'\n';

        return TRUE;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime);

    int BackPropagateOnGPU(int pTime);

#endif  // __CUDNN__

};


#endif  // Embedding_H_
