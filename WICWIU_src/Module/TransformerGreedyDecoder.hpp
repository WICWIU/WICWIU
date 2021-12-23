#ifndef TRANSFORMERGREEDYDECODER_HPP_
#define TRANSFORMERGREEDYDECODER_HPP_

#include "../Module.hpp"

template <typename DTYPE> class TransformerGreedyDecoder : public Module<DTYPE> {
private:
      Operator<DTYPE> *m_pDecoderInput;
      TransformerDecoderModule<DTYPE> *m_pDecoder;

      int m_maxTime;
      int m_endTok;



public:
  TransformerGreedyDecoder(Operator<DTYPE> *pInput,Operator<DTYPE> *pContext, Operator<DTYPE> *pSrcMask, Operator<DTYPE> *pTgtMask, int nLayer, int vocabSize, int vocabLength, int embeddingDim, int nHead, int endTok, std::string Name) : Module<DTYPE>(Name) {
      #ifdef __DEBUG__
      std::cout << "TransformerGreedyDecoder(Operator<DTYPE> *,Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, int , int , int , int , int , int , std::string\n";
      #endif  // __DEBUG__

      m_pDecoderInput = NULL;
      m_pDecoder = NULL;
      m_maxTime = 0;
      m_endTok = 0;

      Alloc(pInput, pContext, pSrcMask, pTgtMask, nLayer, vocabSize, vocabLength, embeddingDim, nHead, endTok, Name);
  }

  int Alloc(Operator<DTYPE> *pInput,Operator<DTYPE> *pContext, Operator<DTYPE> *pSrcMask, Operator<DTYPE> *pTgtMask, int nLayer, int vocabSize, int vocabLength, int embeddingDim, int nHead, int endTok, std::string Name) {
      #ifdef __DEBUG__
      std::cout << "TransformerGreedyDecoder(Operator<DTYPE> *,Operator<DTYPE> *, Operator<DTYPE> *, Operator<DTYPE> *, int , int , int , int , int , int , std::string\n";
      #endif  // __DEBUG__
      this->SetInput(4, pInput, pContext, pSrcMask, pTgtMask);

      m_pDecoderInput = pInput;

      m_maxTime = vocabLength;
      m_endTok = endTok;

      // Tensor<DTYPE> *pInputResult = pInput->GetResult();
      // Shape *pShape = pInputResult->GetShape();
      //
      // int timeSize = (*pShape)[0];
      // int batchSize = (*pShape)[1];
      // int channelSize = (*pShape)[2];
      // int rowSize = (*pShape)[3];
      // int colSize = (*pShape)[4];

      // m_pConnectingTensorHolder = new Tensorholder<DTYPE>(timeSize, batchSize, channelSize, rowSize, colSize, "GreedyDecoderConnectingOp");
      // m_pConnectingTensorHolder->SetIsTensorholder(FALSE);

      m_pDecoder = new TransformerDecoderModule<DTYPE>(m_pDecoderInput, pContext, pSrcMask, pTgtMask, nLayer, vocabSize, vocabLength, embeddingDim, nHead, Name);

      this->AnalyzeGraph(m_pDecoder);

      return TRUE;
  }

  int ForwardPropagate(int pTime = 0) {

      // Tensor<DTYPE> *pDecoderInput = m_pDecoderInput->GetResult();
      // Shape *pShape = pDecoderInput->GetShape();
      //
      // int resultVocabIdx[m_maxTime] = {1,};
      //
      // std::cout<<"inputTen = \n";
      // std::cout<<pDecoderInput<<"\n";
      //
      // // std::cout<<"Forward Called\n";
      //
      // for(int i=0; i<m_maxTime; i++) {
      //     m_pDecoder->ForwardPropagate(pTime);
      //     // std::cout<<"Decoder Forward Out\n";
      //
      //     int inference = m_pDecoder->GetResult()->Argmax(4, i, 0);
      //     resultVocabIdx[i+1] = inference;
      //     for(int j=0; j<=i+1; j++)
      //         (*pDecoderInput)[Index5D(pShape, 0, 0, 0, 0, j)] = (float)resultVocabIdx[j];
      //
      //     if(inference == m_endTok) break;
      // }


      return TRUE;
  }

  int BackPropagate(int pTime = 0) {
      return TRUE;
  }

  #ifdef __CUDNN__
      int ForwardPropagateOnGPU() {
          Tensor<DTYPE> *pDecoderInput = m_pDecoderInput->GetResult();

          Shape *pShape = pDecoderInput->GetShape();

          return TRUE;
      }

      int BackPropagateOnGPU() {
          return TRUE;
      }
  #endif // __CUDNN__
};




#endif
