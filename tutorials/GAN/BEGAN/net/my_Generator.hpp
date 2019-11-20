#include <iostream>
#include <string>

#include "../../../../WICWIU_src/NeuralNetwork.hpp"

using namespace std;

template<typename DTYPE> class my_Generator : public NeuralNetwork<DTYPE> {
private:
public:
    my_Generator(Operator<float> *z){
        Alloc(z);
    }

    virtual ~my_Generator() {
    }

    int Alloc(Operator<float> *z){
        this->SetInput(z);

        Operator<float> *out = z;
        const int D = 256;
        // for 64x64
        // ======================= layer 1 ======================
        out = new Linear<float>(out, 100, 4 * 4 * D*8);
        out = new Relu<float>(out, "G_R1");
        out = new ReShape<float>(out, D*8, 4, 4, "plz");
        // ======================= layer 2 ======================
        out = new TransposedConvolutionLayer2D<float>(out, D*8, D*4, 4, 4, 2, 2, 1, 1, "G_TransposedConv2");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN2");
        out = new Relu<float>(out, "G_R2");
        // ======================= layer 3 ======================
        out = new TransposedConvolutionLayer2D<float>(out, D*4, D*2, 4, 4, 2, 2, 1, 1, "G_TransposedConv3");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN3");
        out = new Relu<float>(out, "G_R3");
        // ======================= layer 4 ======================
        out = new TransposedConvolutionLayer2D<float>(out, D*2, D, 4, 4, 2, 2, 1, 1, "G_TransposedConv4");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN4");
        out = new Relu<float>(out, "G_R4");
        // ======================= layer 5 ====================
        out = new TransposedConvolutionLayer2D<float>(out, D, 3, 4, 4, 2, 2, 1, 1, "G_TransposedConv5");
        out = new Tanh<float>(out, "G_Tanh1");

        out = new ReShape<float>(out, 1, 1, 3 * 64 * 64, "Img2Flat");

        this->AnalyzeGraph(out);
    }
};
