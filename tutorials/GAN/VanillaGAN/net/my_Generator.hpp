#include <iostream>
#include <string>

#include "../../../../WICWIU_src/NeuralNetwork.hpp"

template <typename DTYPE>
class my_Generator : public NeuralNetwork<DTYPE>
{
public:
    my_Generator(Operator<float>* z) { Alloc(z); }

    virtual ~my_Generator() {}

    int Alloc(Operator<float>* z)
    {
        this->SetInput(z);

        Operator<float>* out = z;

        // ======================= layer 1 ======================
        out = new Linear<float>(out, 100, 128, TRUE, "G_L1");

        // ======================= layer 2 ======================
        out = new Linear<float>(out, 128, 256, TRUE, "G_L2");
        out = new BatchNormalizeLayer<DTYPE>(out, FALSE, "G_BN1");
        out = new LRelu<float>(out, 0.2, "G_LRelu1");
        // out = new Relu<float>(out, "G_Relu1");

        // ======================= layer 3 ======================
        out = new Linear<float>(out, 256, 512, TRUE, "G_L3");
        out = new BatchNormalizeLayer<DTYPE>(out, FALSE, "G_BN2");
        out = new LRelu<float>(out, 0.2, "G_LRelu2");
        // out = new Relu<float>(out, "G_Relu2");

        // ======================= layer 4 ======================
        out = new Linear<float>(out, 512, 1024, TRUE, "G_L4");
        out = new BatchNormalizeLayer<DTYPE>(out, FALSE, "G_BN3");
        out = new LRelu<float>(out, 0.2, "G_LRelu3");
        // out = new Relu<float>(out, "G_Relu3");

        // ======================= layer 5 ====================
        out = new Linear<float>(out, 1024, 784, TRUE, "G_L5");
        out = new Tanh<float>(out, "Tanh");

        this->AnalyzeGraph(out);
    }
};
