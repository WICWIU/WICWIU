#include "../../../WICWIU_src/NeuralNetwork.h"

template<typename DTYPE> class BasicBlock : public Module<DTYPE>{
private:
public:
    BasicBlock(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOutputChannel, int pStride = 1, std::string pName = NULL) {
        Alloc(pInput, pNumInputChannel, pNumOutputChannel, pStride, pName);
    }

    virtual ~BasicBlock() {}

    int Alloc(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOutputChannel, int pStride, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *remember = pInput;
        Operator<DTYPE> *out      = pInput;

        // 1
        out = new ConvolutionLayer2D<DTYPE>(out, pNumInputChannel, pNumOutputChannel, 3, 3, pStride, pStride, 1, FALSE, "BasicBlock_Conv1" + pName);
        out = new BatchNormalizeLayer<float>(out, TRUE, "BasicBlock_BN_1");
        out = new Relu<DTYPE>(out, "BasicBlock_Relu1" + pName);

        // 2
        out = new ConvolutionLayer2D<DTYPE>(out, pNumOutputChannel, pNumOutputChannel, 3, 3, 1, 1, 1, FALSE, "BasicBlock_Conv2" + pName);
        out = new BatchNormalizeLayer<float>(out, TRUE, "BasicBlock_BN_2");

        // ShortCut
        if ((pStride != 1) || (pNumInputChannel != pNumOutputChannel)) {
            remember = new ConvolutionLayer2D<DTYPE>(remember, pNumInputChannel, pNumOutputChannel, 3, 3, pStride, pStride, 1, FALSE, "BasicBlock_Conv_Shortcut" + pName);
            out      = new BatchNormalizeLayer<float>(out, TRUE, "BasicBlock_BN_shortcut");
        }

        // Add (for skip Connection)
        out = new Addall<DTYPE>(remember, out, "ResNet_Skip_Add" + pName);
        // out = new Addall<DTYPE>(remember, out, "ResNet_Skip_Add");

        // Last Relu
        out = new Relu<DTYPE>(out, "BasicBlock_Relu2" + pName);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

// template<typename DTYPE> class Bottleneck : public Module<DTYPE>{
// private:
// int m_expansion;
//
// public:
// Bottleneck(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOfChannel, int pStride = 1, int pExpansion = 1) {
// Alloc(pInput, pNumInputChannel, pNumOfChannel, pStride, pExpansion);
// }
//
// virtual ~Bottleneck() {}
//
// int Alloc(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOfChannel, int pStride, int pExpansion) {
// m_expansion = pExpansion;
//
// Operator<DTYPE> *remember = pInput;
// Operator<DTYPE> *out      = pInput;
//
//// 1
// out =new ConvolutionLayer2D<DTYPE>(out, pNumInputChannel, pNumOfChannel, 1, 1, pStride, pStride, 1, FALSE, "BasicBlock_Conv1");
// out =new BatchNormalizeLayer2D<DTYPE>(out, pNumOfChannel, "BasicBlock_BN1");
//
// out =new Relu<DTYPE>(out, "BasicBlock_Relu1");
//
//// 2
// out =new ConvolutionLayer2D<DTYPE>(out, pNumOfChannel, pNumOfChannel, 3, 3, 1, 1, 1, FALSE, "BasicBlock_Conv1");
// out =new BatchNormalizeLayer2D<DTYPE>(out, pNumOfChannel, "BasicBlock_BN1");
//
// out =new Relu<DTYPE>(out, "BasicBlock_Relu1");
//
//// 3
// out =new ConvolutionLayer2D<DTYPE>(out, pNumOfChannel, m_expansion * pNumOfChannel, 3, 3, 1, 1, 1, FALSE, "BasicBlock_Conv1");
// out =new BatchNormalizeLayer2D<DTYPE>(out, m_expansion * pNumOfChannel, "BasicBlock_BN1");
//
//// ShortCut
// if ((pStride != 1) || (pNumInputChannel != m_expansion * pNumOfChannel)) {
// remember =new ConvolutionLayer2D<DTYPE>(remember, pNumInputChannel, m_expansion * pNumOfChannel, 3, 3, pStride, pStride, 1, FALSE, "BasicBlock_Conv1");
// remember =new BatchNormalizeLayer2D<DTYPE>(remember, m_expansion * pNumOfChannel, "BasicBlock_BN1");
// }
//
// out =new Addall<DTYPE>(remember, out, "ResNet_Skip_Add");
//// out =new Addall<DTYPE>(remember, "ResNet_Skip_Add");
//
// out =new Relu<DTYPE>(out, "BasicBlock_Relu1");
//
// return TRUE;
// }
// };

template<typename DTYPE> class ResNet : public NeuralNetwork<DTYPE>{
private:
    int m_numInputChannel;

public:
    ResNet(Tensorholder<DTYPE> *pInput, Tensorholder<DTYPE> *pLabel, std::string pBlockType, int pNumOfBlock1, int pNumOfBlock2, int pNumOfBlock3, int pNumOfBlock4, int pNumOfClass) {
        Alloc(pInput, pLabel, pBlockType, pNumOfBlock1, pNumOfBlock2, pNumOfBlock3, pNumOfBlock4, pNumOfClass);
    }

    virtual ~ResNet() {}

    int Alloc(Tensorholder<DTYPE> *pInput, Tensorholder<DTYPE> *pLabel, std::string pBlockType, int pNumOfBlock1, int pNumOfBlock2, int pNumOfBlock3, int pNumOfBlock4, int pNumOfClass) {
        this->SetInput(2, pInput, pLabel);

        m_numInputChannel = 64;

        Operator<DTYPE> *out = pInput;

        // ReShape
        out = new ReShape<DTYPE>(out, 28, 28, "ReShape");

        // out = new BatchNormalizeLayer<float>(out, TRUE, "BN_0");

        // 1
        out = new ConvolutionLayer2D<DTYPE>(out, 1, m_numInputChannel, 1, 1, 1, 1, 0, TRUE, "Conv");
        out = new BatchNormalizeLayer<float>(out, TRUE, "BN_1");

        out = this->MakeLayer(out, m_numInputChannel, pBlockType, pNumOfBlock1, 1, "Block1");
        out = this->MakeLayer(out, 128, pBlockType, pNumOfBlock2, 1, "Block2");
        out = this->MakeLayer(out, 256, pBlockType, pNumOfBlock3, 2, "Block3");
        out = this->MakeLayer(out, 512, pBlockType, pNumOfBlock3, 2, "Block4");

        out = new GlobalAvaragePooling2D<DTYPE>(out, "Avg Pooling");

        out = new ReShape<DTYPE>(out, 1, 1, 512, "ReShape");

        out = new Linear<DTYPE>(out, 512, pNumOfClass, TRUE, "Classification");

        out = new BatchNormalizeLayer<float>(out, FALSE, "BN_0");

        this->AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        this->SetLossFunction(new SoftmaxCrossEntropy<float>(out, pLabel, "SCE"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================
        this->SetOptimizer(new GradientDescentOptimizer<float>(this->GetParameter(), 0.1, 0.9, 5e-4, MINIMIZE));
        // this->SetOptimizer(new GradientDescentOptimizer<float>(this->GetParameter(), 0.001, MINIMIZE));

        return TRUE;
    }

    Operator<DTYPE>* MakeLayer(Operator<DTYPE> *pInput, int pNumOfChannel, std::string pBlockType, int pNumOfBlock, int pStride, std::string pName = NULL) {
        if (pNumOfBlock == 0) {
            return pInput;
        } else if ((pBlockType == "BasicBlock") && (pNumOfBlock > 0)) {
            Operator<DTYPE> *out = pInput;

            // Test of effect of the Max pool
            if (pStride > 1) {
                out = new Maxpooling2D<float>(out, pStride, pStride, 2, 2, "MaxPool_2");
            }

            out = new BasicBlock<DTYPE>(out, m_numInputChannel, pNumOfChannel, 1, pName);

            int pNumOutputChannel = pNumOfChannel;

            // int pNumOutputChannel = pNumOfChannel;
            //
            // out =new BasicBlock<DTYPE>(out, m_numInputChannel, pNumOutputChannel, pStride, pName);

            for (int i = 1; i < pNumOfBlock; i++) {
                out = new BasicBlock<DTYPE>(out, pNumOutputChannel, pNumOutputChannel, 1, pName);
            }

            m_numInputChannel = pNumOutputChannel;

            return out;
        } else if ((pBlockType == "Bottleneck") && (pNumOfBlock > 0)) {
            return NULL;
        } else return NULL;
    }
};

template<typename DTYPE> NeuralNetwork<DTYPE>* Resnet14(Tensorholder<DTYPE> *pInput, Tensorholder<DTYPE> *pLabel) {
    return new ResNet<DTYPE>(pInput, pLabel, "BasicBlock", 2, 2, 2, 2, 10);
}
