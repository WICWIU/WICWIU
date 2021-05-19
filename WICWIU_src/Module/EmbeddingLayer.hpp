#ifndef __EMBEDDING_LAYER__
#define __EMBEDDING_LAYER__    value

#include "../Module.hpp"


template<typename DTYPE> class EmbeddingLayer : public Module<DTYPE>{
private:
public:
    /*!
    @brief CBOWLayer 클래스 생성자
    @details CBOWLayer 클래스의 Alloc 함수를 호출한다.*/
    EmbeddingLayer(Operator<DTYPE> *pInput, int vocabsize, int embeddingDim, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, vocabsize, embeddingDim, pName);
    }

    /*!
    @brief CBOWLayer 클래스 소멸자
    @details 단, 동적 할당 받은 Operator들은 NeuralNetwork에서 할당 해제한다.  */
    virtual ~EmbeddingLayer() {}


    int Alloc(Operator<DTYPE> *pInput, int vocabsize, int embeddingDim, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;


        // std::cout<<"SKIPGRAMLayer vocabsize : "<<vocabsize<<'\n';

        //------------------------------weight 생성-------------------------
        //Win 여기서 window 사이즈만큼 곱하기 안해주는 이유 : input에서 잘라서 값 복사해서 처리해주기???
        Tensorholder<DTYPE> *pWeight_embedding = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, vocabsize, embeddingDim, 0.0, 0.01), "EmbedLayer_pWeight" + pName);


        out = new Embedding<DTYPE>(pWeight_embedding, out, "Embedding_Operator");

        this->AnalyzeGraph(out);

        return TRUE;
    }

};


#endif  // __SKIPGRAM_LAYER__
