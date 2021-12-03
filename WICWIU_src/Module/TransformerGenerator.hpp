#ifndef TRANSFORMERGENERATOR_HPP_
#define TRANSFORMERGENERATOR_HPP_

#include "../Module.hpp"


template<typename DTYPE> class TransformerGenerator : public Module<DTYPE> {
private:
public:
  TransformerGenerator(Operator<DTYPE> *pInput, int d_model, int vocabSize, std::string pName = "NO NAME") : Module<DTYPE>(pName) {
    #ifdef __DEBUG__
    std::cout << "TransformerGenerator::TransformerGenerator(Operator<DTYPE> *, int , int , std::string )" << '\n';
    #endif  // __DEBUG__

    Alloc(pInput, d_model, vocabSize, pName);
  }

  int Alloc(Operator<DTYPE> *pInput, int d_model, int vocabSize, std::string pName) {
    #ifdef __DEBUG__
    std::cout << "TransformerGenerator::Alloc(Operator<DTYPE> *, int , int , std::string )" << '\n';
    #endif  // __DEBUG__

    this->SetInput(1, pInput);

    Operator<DTYPE> *out = NULL;

    out = new Linear<DTYPE>(pInput, d_model, vocabSize, FALSE, pName+"LinearLayer");
    out = new Softmax1D<DTYPE>(out, 1e-6f, 4, pName+"Softmax");

    this->AnalyzeGraph(out);

    return TRUE;
  }
};





#endif
