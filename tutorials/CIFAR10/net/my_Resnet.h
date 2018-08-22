#include "../../../WICWIU_src/NeuralNetwork.h"

template<typename DTYPE> class ResnetBasicBlock : public Module<DTYPE>{
private:
public:
    ResnetBasicBlock(Operator<DTYPE> *input, int numOfInputChannel, int numOfOutputChannel, int stride = 1, std::string name = NULL) {
        Alloc(input, numOfInputChannel, numOfOutputChannel, stride, name);
    }

    virtual ~ResnetBasicBlock() {}

    int Alloc(Operator<DTYPE> *input, int numOfInputChannel, int numOfOutputChannel, int stride, std::string name) {
        this->SetInput(input);

        Operator<DTYPE> *remember = input;
        Operator<DTYPE> *out      = input;

        // 1st Layer
        out = new ConvolutionLayer2D<DTYPE>(out, numOfInputChannel, numOfOutputChannel, 3, 3, stride, stride, 1, FALSE, "Conv1" + name);
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN1" + name);
        out = new Relu<DTYPE>(out, "Relu1" + name);

        // 2nd Layer
        out = new ConvolutionLayer2D<DTYPE>(out, numOfOutputChannel, numOfOutputChannel, 3, 3, 1, 1, 1, FALSE, "Conv2" + name);
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN2" + name);

        // ShortCut
        if ((stride != 1) || (numOfInputChannel != numOfOutputChannel)) {
            remember = new ConvolutionLayer2D<DTYPE>(remember, numOfInputChannel, numOfOutputChannel, 3, 3, stride, stride, 1, FALSE, "Conv_Shortcut" + name);
            remember = new BatchNormalizeLayer<DTYPE>(remember, TRUE, "BN_Shortcut" + name);
        }

        // Add (for skip Connection)
        out = new Addall<DTYPE>(remember, out, "Add_for_skip_connection" + name);

        // Activation Function
        out = new Relu<DTYPE>(out, "Relu2" + name);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

template<typename DTYPE> class ResnetBottleneckBlock : public Module<DTYPE>{
private:
    int m_expansion;

public:
    ResnetBottleneckBlock(Operator<DTYPE> *input, int numOfInputChannel, int numOfOutputChannel, int stride = 1, int expansion = 1, std::string name = NULL) {
        Alloc(input, numOfInputChannel, numOfOutputChannel, stride, expansion, name);
    }

    virtual ~ResnetBottleneckBlock() {}

    int Alloc(Operator<DTYPE> *input, int numOfInputChannel, int numOfOutputChannel, int stride, int expansion, std::string name) {
        m_expansion = expansion;

        Operator<DTYPE> *remember = input;
        Operator<DTYPE> *out      = input;

        // 1st Layer
        out = new ConvolutionLayer2D<DTYPE>(out, numOfInputChannel, numOfOutputChannel, 1, 1, stride, stride, 1, FALSE, "Conv1" + name);
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN1" + name);
        out = new Relu<DTYPE>(out, "Relu1" + name);

        // 2nd Layer
        out = new ConvolutionLayer2D<DTYPE>(out, numOfOutputChannel, numOfOutputChannel, 3, 3, 1, 1, 1, FALSE, "Conv2" + name);
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN2" + name);
        out = new Relu<DTYPE>(out, "Relu2" + name);

        // 3rd Layer
        out = new ConvolutionLayer2D<DTYPE>(out, numOfOutputChannel, m_expansion * numOfOutputChannel, 3, 3, 1, 1, 1, FALSE, "Conv3" + name);
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN3" + name);

        // ShortCut
        if ((stride != 1) || (numOfInputChannel != m_expansion * numOfOutputChannel)) {
            remember = new ConvolutionLayer2D<DTYPE>(remember, numOfInputChannel, m_expansion * numOfOutputChannel, 3, 3, stride, stride, 1, FALSE, "Conv_Shortcut" + name);
            remember = new BatchNormalizeLayer<DTYPE>(remember, TRUE, "BN_Shortcut" + name);
        }

        // Add (for skip Connection)
        out = new Addall<DTYPE>(remember, out, "Add_for_skip_connection" + name);

        // Activation Function
        out = new Relu<DTYPE>(out, "Relu3" + name);

        return TRUE;
    }
};

template<typename DTYPE> class ResNet : public NeuralNetwork<DTYPE>{
private:
    int m_numOfInputChannel;

public:
    ResNet(Tensorholder<DTYPE> *input, Tensorholder<DTYPE> *label, std::string typeOfBlock, int numOfBlock1, int numOfBlock2, int numOfBlock3, int numOfBlock4, int numOfClass) {
        Alloc(input, label, typeOfBlock, numOfBlock1, numOfBlock2, numOfBlock3, numOfBlock4, numOfClass);
    }

    virtual ~ResNet() {}

    int Alloc(Tensorholder<DTYPE> *input, Tensorholder<DTYPE> *label, std::string typeOfBlock, int numOfBlock1, int numOfBlock2, int numOfBlock3, int numOfBlock4, int numOfClass) {
        this->SetInput(2, input, label);

        m_numOfInputChannel = 64;

        Operator<DTYPE> *out = input;

        // ReShape
        out = new ReShape<DTYPE>(out, 3, 32, 32, "ReShape");

        // 1
        out = new ConvolutionLayer2D<DTYPE>(out, 3, m_numOfInputChannel, 3, 3, 1, 1, 1, FALSE, "Conv");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN1");

        out = this->MakeLayer(out, m_numOfInputChannel, typeOfBlock, numOfBlock1, 1, "Block1");
        out = this->MakeLayer(out, 128, typeOfBlock, numOfBlock2, 2, "Block2");
        out = this->MakeLayer(out, 256, typeOfBlock, numOfBlock3, 2, "Block3");
        out = this->MakeLayer(out, 512, typeOfBlock, numOfBlock3, 2, "Block4");

        out = new GlobalAvaragePooling2D<DTYPE>(out, "Avg Pooling");
        out = new ReShape<DTYPE>(out, 1, 1, 512, "ReShape");

        out = new Linear<DTYPE>(out, 512, numOfClass, TRUE, "Classification");

        this->AnalyzeGraph(out);

        // ======================= Set LossFunction Function ===================
        this->SetLossFunction(new SoftmaxCrossEntropy<float>(out, label, "SCE"));

        // ======================= Set Optimizer ===================
        this->SetOptimizer(new GradientDescentOptimizer<float>(this->GetParameter(), 0.1, 0.9, 5e-4, MINIMIZE));

        return TRUE;
    }

    Operator<DTYPE>* MakeLayer(Operator<DTYPE> *input, int numOfOutputChannel, std::string typeOfBlock, int numOfBlock, int stride, std::string name = NULL) {
        if (numOfBlock == 0) {
            return input;
        } else if ((typeOfBlock == "ResnetBasicBlock") && (numOfBlock > 0)) {
            Operator<DTYPE> *out = input;

            out = new ResnetBasicBlock<DTYPE>(out, m_numOfInputChannel, numOfOutputChannel, stride, name);

            for (int i = 1; i < numOfBlock; i++) {
                out = new ResnetBasicBlock<DTYPE>(out, numOfOutputChannel, numOfOutputChannel, 1, name);
            }

            m_numOfInputChannel = numOfOutputChannel;

            return out;
        } else if ((typeOfBlock == "ResnetBottleneckBlock") && (numOfBlock > 0)) {
            Operator<DTYPE> *out = input;

            out = new ResnetBottleneckBlock<DTYPE>(out, m_numOfInputChannel, numOfOutputChannel, stride, name);

            for (int i = 1; i < numOfBlock; i++) {
                out = new ResnetBottleneckBlock<DTYPE>(out, numOfOutputChannel, numOfOutputChannel, 1, name);
            }

            m_numOfInputChannel = numOfOutputChannel;

            return out;
        } else return NULL;
    }
};

template<typename DTYPE> NeuralNetwork<DTYPE>* Resnet18(Tensorholder<DTYPE> *input, Tensorholder<DTYPE> *label) {
    return new ResNet<DTYPE>(input, label, "BasicBlock", 2, 2, 2, 2, 10);
}

template<typename DTYPE> NeuralNetwork<DTYPE>* Resnet34(Tensorholder<DTYPE> *input, Tensorholder<DTYPE> *label) {
    return new ResNet<DTYPE>(input, label, "BasicBlock", 3, 4, 6, 3, 10);
}
