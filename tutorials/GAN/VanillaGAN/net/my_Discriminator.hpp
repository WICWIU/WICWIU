#include <iostream>
#include <string>

#include "../../../../WICWIU_src/NeuralNetwork.hpp"

template <typename DTYPE>
class my_Discriminator : public NeuralNetwork<DTYPE>
{
public:
    my_Discriminator(Operator<float>* x) { Alloc(x); }

    virtual ~my_Discriminator() {}

    int Alloc(Operator<float>* x)
    {
        this->SetInput(x);

        Operator<float>* out = x;

        // ======================= layer 1 ======================
        out = new Linear<float>(out, 784, 512, TRUE, "D_L1");
        out = new LRelu<float>(out, 0.2, "D_LRelu1");
        // out = new Relu<float>(out, "D_Relu1");

        // ======================= layer 2 ======================
        out = new Linear<float>(out, 512, 256, TRUE, "D_L2");
        out = new LRelu<float>(out, 0.2, "D_LRelu2");
        // out = new Relu<float>(out, "D_Relu2");

        // ======================= layer 3 ======================
        out = new Linear<float>(out, 256, 1, TRUE, "D_L3");
        out = new Sigmoid<float>(out, "D_Sigmoid");

        this->AnalyzeGraph(out);
    }
};
