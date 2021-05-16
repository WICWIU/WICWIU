#ifndef __DECODER__
#define __DECODER__    value

#include "../Module.hpp"


template<typename DTYPE> class Decoder : public Module<DTYPE>{
private:

    int timesize;

    Operator<DTYPE> *m_initHiddenTensorholder;

    Operator<DTYPE> *m_EncoderLengths;

    int m_isTeacherForcing;

public:

    Decoder(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, int vocabLength, int embeddingDim, int hiddenSize, int outputSize, int m_isTeacherForcing = TRUE, Operator<DTYPE> *pEncoderLengths = NULL, int useBias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, pEncoder, vocabLength, embeddingDim, hiddenSize, outputSize, m_isTeacherForcing, pEncoderLengths, useBias, pName);
    }


    virtual ~Decoder() {}


    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pEncoder, int vocabLength, int embeddingDim, int hiddenSize, int outputSize, int teacherForcing, Operator<DTYPE> *pEncoderLengths, int useBias, std::string pName) {

        this->SetInput(2, pInput, pEncoder);

        timesize = pInput->GetResult()->GetTimeSize();
        int batchsize = pInput->GetResult()->GetBatchSize();
        m_initHiddenTensorholder  = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(1, batchsize, 1, 1, hiddenSize), "tempHidden");

        m_EncoderLengths = pEncoderLengths;
        m_isTeacherForcing = teacherForcing;

        Operator<DTYPE> *out = pInput;

        out = new EmbeddingLayer<DTYPE>(out, vocabLength, embeddingDim, "Embedding");

         // out = new RecurrentLayer<DTYPE>(out, embeddingDim, hiddenSize, m_initHiddenTensorholder, useBias, "Recur_1");
        out = new LSTMLayer<DTYPE>(out, embeddingDim, hiddenSize, m_initHiddenTensorholder, useBias, "Recur_1");
        // out = new GRULayer<DTYPE>(out, embeddingDim, hiddenSize, m_initHiddenTensorholder, useBias, "Recur_1");


        out = new Linear<DTYPE>(out, hiddenSize, outputSize, useBias, "Fully-Connected-H2O");

        this->AnalyzeGraph(out);

        return TRUE;
    }


    int ForwardPropagate(int pTime=0) {

        if(pTime == 0){
              Tensor<DTYPE> *_initHidden = this->GetInput()[1]->GetResult();
              Tensor<DTYPE> *initHidden = m_initHiddenTensorholder->GetResult();

              Shape *_initShape = _initHidden->GetShape();
              Shape *initShape = initHidden->GetShape();

              int enTimesize = _initHidden->GetTimeSize();
              int batchsize  = _initHidden->GetBatchSize();
              int colSize    = _initHidden->GetColSize();

              if( m_EncoderLengths != NULL){

                  Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();

                  for(int ba=0; ba<batchsize; ba++){
                      for(int co=0; co<colSize; co++){
                          (*initHidden)[Index5D(initShape, 0, ba, 0, 0, co)] = (*_initHidden)[Index5D(_initShape, (*encoderLengths)[ba]-1, ba, 0, 0, co)];
                      }
                  }
              }else{

                  for(int ba=0; ba<batchsize; ba++){
                      for(int co=0; co<colSize; co++){
                          (*initHidden)[Index5D(initShape, 0, ba, 0, 0, co)] = (*_initHidden)[Index5D(_initShape, enTimesize-1, ba, 0, 0, co)];
                      }
                  }
            }
        }

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = 0; i < numOfExcutableOperator; i++)
            (*ExcutableOperator)[i]->ForwardPropagate(pTime);

        return TRUE;
    }


    int BackPropagate(int pTime=0) {

        int numOfExcutableOperator = this->GetNumOfExcutableOperator();
        Container<Operator<DTYPE> *> *ExcutableOperator = this->GetExcutableOperatorContainer();

        for (int i = numOfExcutableOperator - 1; i >= 0; i--) {
            (*ExcutableOperator)[i]->BackPropagate(pTime);
        }

        if(pTime == 0){
              Tensor<DTYPE> *enGradient = this->GetInput()[1]->GetGradient();
              Tensor<DTYPE> *_enGradient = m_initHiddenTensorholder->GetGradient();

              Shape *enShape  = enGradient->GetShape();
              Shape *_enShape = _enGradient->GetShape();

              int enTimesize = enGradient->GetTimeSize();
              int batchSize = enGradient->GetBatchSize();
              int colSize = enGradient->GetColSize();


              if( m_EncoderLengths != NULL){

                  Tensor<DTYPE> *encoderLengths = m_EncoderLengths->GetResult();

                  for(int ba=0; ba < batchSize; ba++){
                      for(int co=0; co < colSize; co++){
                          (*enGradient)[Index5D(enShape, (*encoderLengths)[ba]-1, ba, 0, 0, co)] = (*_enGradient)[Index5D(_enShape, 0, ba, 0, 0, co)];
                      }
                  }

              }
              else{
                  for(int ba=0; ba < batchSize; ba++){
                      for(int co=0; co < colSize; co++){
                          (*enGradient)[Index5D(enShape, enTimesize-1, ba, 0, 0, co)] = (*_enGradient)[Index5D(_enShape, 0, ba, 0, 0, co)];
                      }
                  }
              }

        }

        return TRUE;
    }


  #ifdef __CUDNN__
      int ForwardPropagateOnGPU(int pTime = 0);
      int BackPropagateOnGPU(int pTime = 0);
  #endif // CUDNN

};

#endif  // __DECODER__
