#include <iostream>
#include <string>

#include "../../../../WICWIU_src/NeuralNetwork.hpp"
using namespace std;

template <typename DTYPE>
class my_Discriminator : public NeuralNetwork<DTYPE>
{
private:
public:
    my_Discriminator(Operator<float>* x) { Alloc(x); }

    virtual ~my_Discriminator() {}

    int Alloc(Operator<float>* x)
    {
        this->SetInput(x);

        const int D = 256;

        Operator<float>* out = x;
        Operator<float>* input = x;

        // for 64 x 64
        out = new ReShape<float>(out, 3, 64, 64, "Flat2Img");
        // ======================= layer 1 ======================
        out = new ConvolutionLayer2D<float>(out, 3, D * 2, 4, 4, 2, 2, 1, 1, "D_Conv1");
        out = new LRelu<float>(out, 0.2, "D_R1");
        // ======================= layer 2 ======================
        out = new ConvolutionLayer2D<float>(out, D * 2, D, 4, 4, 2, 2, 1, 1, "D_Conv2");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN1");
        out = new LRelu<float>(out, 0.2, "D_R2");
        // ======================= layer 3 ======================
        out = new ConvolutionLayer2D<float>(out, D, 8, 4, 4, 2, 2, 1, 1, "D_Conv3");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN2");
        out = new LRelu<float>(out, 0.2, "D_R3");

        out = new ReShape<float>(out, 1, 1, 8 * 8 * 8, "plz");

        // ======================= layer 1 ======================
        out = new Linear<float>(out, 8 * 8 * 8, 100);
        out = new Linear<float>(out, 100, 8 * 8 * 8);

        out = new ReShape<float>(out, 8, 8, 8, "plz");

        // ======================= layer 3 ======================
        out = new TransposedConvolutionLayer2D<float>(out, 8, D * 2, 4, 4, 2, 2, 1, 1,
                                                      "G_TransposedConv3");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN3");
        out = new Relu<float>(out, "G_R3");
        // ======================= layer 4 ======================
        out = new TransposedConvolutionLayer2D<float>(out, D * 2, D, 4, 4, 2, 2, 1, 1,
                                                      "G_TransposedConv4");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN4");
        out = new Relu<float>(out, "G_R4");
        // ======================= layer 5 ====================
        out = new TransposedConvolutionLayer2D<float>(out, D, 3, 4, 4, 2, 2, 1, 1,
                                                      "G_TransposedConv5");
        out = new Tanh<float>(out, "G_Tanh1");

        out = new ReShape<float>(out, 1, 1, 3 * 64 * 64, "Img2Flat");

        out = new ReconstructionError<float>(out, input, "ReconstructionError");

        this->AnalyzeGraph(out);
    }
};
