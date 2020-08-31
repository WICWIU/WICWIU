#include "../../../WICWIU_src/NeuralNetwork.hpp"

template <typename DTYPE>
class BasicBlock : public Module<DTYPE>
{
private:
public:
    BasicBlock(Operator<DTYPE>* pInput, int pNumInputChannel, int pNumOutputChannel,
               int pStride = 1, std::string pName = NULL)
    {
        Alloc(pInput, pNumInputChannel, pNumOutputChannel, pStride, pName);
    }

    virtual ~BasicBlock() {}

    int Alloc(Operator<DTYPE>* pInput, int pNumInputChannel, int pNumOutputChannel, int pStride,
              std::string pName)
    {
        this->SetInput(pInput);

        Operator<DTYPE>* remember = pInput;
        Operator<DTYPE>* out = pInput;

        // 1
        out = new ConvolutionLayer2D<DTYPE>(out, pNumInputChannel, pNumOutputChannel, 3, 3, pStride,
                                            pStride, 1, FALSE, "BasicBlock_Conv1" + pName);
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BasicBlock_BN1" + pName);
        out = new Relu<DTYPE>(out, "BasicBlock_Relu1" + pName);

        // 2
        out = new ConvolutionLayer2D<DTYPE>(out, pNumOutputChannel, pNumOutputChannel, 3, 3, 1, 1,
                                            1, FALSE, "BasicBlock_Conv2" + pName);
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BasicBlock_BN2" + pName);
        // out = new Relu<DTYPE>(out, "BasicBlock_Relu2" + pName);

        // ShortCut
        if ((pStride != 1) || (pNumInputChannel != pNumOutputChannel))
        {
            remember = new ConvolutionLayer2D<DTYPE>(remember, pNumInputChannel, pNumOutputChannel,
                                                     1, 1, pStride, pStride, 0, FALSE,
                                                     "BasicBlock_Conv3_Shortcut" + pName);
            remember =
                new BatchNormalizeLayer<DTYPE>(remember, TRUE, "BasicBlock_BN3_Shortcut" + pName);
            // remember = new Relu<DTYPE>(remember, "BasicBlock_Relu3_Shortcut" + pName);
        }

        // Add (for skip Connection)
        out = new Addall<DTYPE>(remember, out, "ResNet_Skip_Add" + pName);
        // out = new Addall<DTYPE>(remember, out, "ResNet_Skip_Add");

        // Last Relu
        out = new Relu<DTYPE>(out, "BasicBlock_Relu3" + pName);

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

template <typename DTYPE>
class ResNet : public NeuralNetwork<DTYPE>
{
private:
    int m_numInputChannel;

public:
    ResNet(Tensorholder<DTYPE>* pInput, Tensorholder<DTYPE>* pLabel, std::string pBlockType,
           int pNumOfBlock1, int pNumOfBlock2, int pNumOfBlock3, int pNumOfBlock4, int pNumOfClass)
    {
        Alloc(pInput, pLabel, pBlockType, pNumOfBlock1, pNumOfBlock2, pNumOfBlock3, pNumOfBlock4,
              pNumOfClass);
    }

    virtual ~ResNet() {}

    int Alloc(Tensorholder<DTYPE>* pInput, Tensorholder<DTYPE>* pLabel, std::string pBlockType,
              int pNumOfBlock1, int pNumOfBlock2, int pNumOfBlock3, int pNumOfBlock4,
              int pNumOfClass)
    {
        this->SetInput(2, pInput, pLabel);

        m_numInputChannel = 64;

        Operator<DTYPE>* out = pInput;

        // ReShape
        out = new ReShape<DTYPE>(out, 3, 224, 224, "ReShape");
        // out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN0");

        // 1
        out =
            new ConvolutionLayer2D<DTYPE>(out, 3, m_numInputChannel, 7, 7, 2, 2, 3, FALSE, "Conv");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN0");
        out = new Relu<DTYPE>(out, "Relu0");

        out = new Maxpooling2D<float>(out, 3, 3, 2, 2, 1, "MaxPool_2");
        // out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN1");

        out = this->MakeLayer(out, m_numInputChannel, pBlockType, pNumOfBlock1, 1, "Block1");
        out = this->MakeLayer(out, 128, pBlockType, pNumOfBlock2, 2, "Block2");
        out = this->MakeLayer(out, 256, pBlockType, pNumOfBlock3, 2, "Block3");
        out = this->MakeLayer(out, 512, pBlockType, pNumOfBlock3, 2, "Block4");

        // out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN1");
        // out = new Relu<DTYPE>(out, "Relu1");

        out = new GlobalAvaragePooling2D<DTYPE>(out, "Avg Pooling");

        out = new ReShape<DTYPE>(out, 1, 1, 512, "ReShape");

        out = new Linear<DTYPE>(out, 512, pNumOfClass, FALSE, "Classification");
        // out = new BatchNormalizeLayer < DTYPE > (out, FALSE, "BN0");

        this->AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        this->SetLossFunction(new SoftmaxCrossEntropy<float>(out, pLabel, "SCE"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================
        // this->SetOptimizer(new GradientDescentOptimizer<float>(this->GetParameter(), 0.1, 0.9,
        // 5e-4, MINIMIZE)); this->SetOptimizer(new
        // GradientDescentOptimizer<float>(this->GetParameter(), 0.001, MINIMIZE));
        this->SetOptimizer(new AdamOptimizer<float>(this->GetParameter(), 0.001, 0.9, 0.999, 1e-08,
                                                    5e-4, MINIMIZE));

        return TRUE;
    }

    Operator<DTYPE>* MakeLayer(Operator<DTYPE>* pInput, int pNumOfChannel, std::string pBlockType,
                               int pNumOfBlock, int pStride, std::string pName = NULL)
    {
        if (pNumOfBlock == 0)
        {
            return pInput;
        }
        else if ((pBlockType == "BasicBlock") && (pNumOfBlock > 0))
        {
            Operator<DTYPE>* out = pInput;

            out = new BasicBlock<DTYPE>(out, m_numInputChannel, pNumOfChannel, pStride, pName);
            int pNumOutputChannel = pNumOfChannel;

            for (int i = 1; i < pNumOfBlock; i++)
            {
                out = new BasicBlock<DTYPE>(out, pNumOutputChannel, pNumOutputChannel, 1, pName);
            }

            m_numInputChannel = pNumOutputChannel;

            return out;
        }
        else if ((pBlockType == "Bottleneck") && (pNumOfBlock > 0))
        {
            return NULL;
        }
        else
            return NULL;
    }
};

template <typename DTYPE>
NeuralNetwork<DTYPE>* Resnet18(Tensorholder<DTYPE>* pInput, Tensorholder<DTYPE>* pLabel,
                               int pNumOfClass)
{
    return new ResNet<DTYPE>(pInput, pLabel, "BasicBlock", 2, 2, 2, 2, pNumOfClass);
}

template <typename DTYPE>
NeuralNetwork<DTYPE>* Resnet34(Tensorholder<DTYPE>* pInput, Tensorholder<DTYPE>* pLabel,
                               int pNumOfClass)
{
    return new ResNet<DTYPE>(pInput, pLabel, "BasicBlock", 3, 4, 6, 3, pNumOfClass);
}
