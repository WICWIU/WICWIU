#ifndef DENSENET_H_
#define DENSENET_H_    0

#include "../../../WICWIU_src/NeuralNetwork.h"

template<typename DTYPE> class DenseNetBlock : public Module<DTYPE>{
private:
public:
    DenseNetBlock(Operator<DTYPE> *pInput, int pNumInputChannel, int pGrowthRate, std::string pName = NULL) : Module<DTYPE>(pName){
        Alloc(pInput, pNumInputChannel, pGrowthRate, pName);
    }

    virtual ~DenseNetBlock() {}

    int Alloc(Operator<DTYPE> *pInput, int pNumInputChannel, int pGrowthRate, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *remember = pInput;
        Operator<DTYPE> *out      = pInput;

        // 1
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "DenseNetBlock_BN1" + pName);
        out = new Relu<DTYPE>(out, "DenseNetBlock_Relu1" + pName);
        out = new ConvolutionLayer2D<DTYPE>(out, pNumInputChannel, 4 * pGrowthRate, 1, 1, 1, 1, 0, FALSE, "DenseNetBlock_Conv1" + pName);

        // 2
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "DenseNetBlock_BN2" + pName);
        out = new Relu<DTYPE>(out, "DenseNetBlock_Relu2" + pName);
        out = new ConvolutionLayer2D<DTYPE>(out, 4 * pGrowthRate, pGrowthRate, 3, 3, 1, 1, 1, FALSE, "DenseNetBlock_Conv2" + pName);

        // Concat
        out = new ConcatenateChannelWise<float>(remember, out, "DenseNetBlock_ConCat");

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

template<typename DTYPE> class Transition : public Module<DTYPE>{
private:
public:
    Transition(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOutputChannel, std::string pName = NULL) : Module<DTYPE>(pName){
        Alloc(pInput, pNumInputChannel, pNumOutputChannel, pName);
    }

    virtual ~Transition() {}

    int Alloc(Operator<DTYPE> *pInput, int pNumInputChannel, int pNumOutputChannel, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;

        // 1
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "Transition_BN" + pName);
        out = new Relu<DTYPE>(out, "Transition_Relu" + pName);
        out = new ConvolutionLayer2D<DTYPE>(out, pNumInputChannel, pNumOutputChannel, 1, 1, 1, 1, 0, FALSE, "Transition_Conv" + pName);

        // Avg Pooling
        out = new AvaragePooling2D<float>(out, 2, 2, 2, 2, 0, "AVG");

        this->AnalyzeGraph(out);

        return TRUE;
    }
};

template<typename DTYPE> class DenseNet : public NeuralNetwork<DTYPE>{
private:
    int m_numInputChannel;
    int m_numOutputChannel;
    int m_growthRate;
    float m_reduction;

public:
    DenseNet(Tensorholder<DTYPE> *pInput, Tensorholder<DTYPE> *pLabel, std::string pBlockType, int pNumOfBlock1, int pNumOfBlock2, int pNumOfBlock3, int pNumOfBlock4, int pGrowthRate = 12, float pReduction = 0.5, int pNumOfClass = 1000) {
        Alloc(pInput, pLabel, pBlockType, pNumOfBlock1, pNumOfBlock2, pNumOfBlock3, pNumOfBlock4, pGrowthRate, pReduction, pNumOfClass);
    }

    virtual ~DenseNet() {}

    int Alloc(Tensorholder<DTYPE> *pInput, Tensorholder<DTYPE> *pLabel, std::string pBlockType, int pNumOfBlock1, int pNumOfBlock2, int pNumOfBlock3, int pNumOfBlock4, int pGrowthRate, float pReduction, int pNumOfClass) {
        this->SetInput(2, pInput, pLabel);

        m_numInputChannel = 2 * pGrowthRate;
        m_growthRate      = pGrowthRate;
        m_reduction       = pReduction;

        Operator<DTYPE> *out = pInput;

        // ReShape
        out = new ReShape<DTYPE>(out, 3, 224, 224, "ReShape");
        // out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN0");

        // 1
        out = new ConvolutionLayer2D<DTYPE>(out, 3, m_numInputChannel, 7, 7, 2, 2, 1, FALSE, "Conv");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN0");
        out = new Relu<DTYPE>(out, "Relu0");

        out                = this->MakeLayer(out, m_numInputChannel, pBlockType, pNumOfBlock1, 1, "Block1");
        m_numInputChannel += pNumOfBlock1 * m_growthRate;
        m_numOutputChannel = (int)(floor(m_numInputChannel * m_reduction));
        out                = new Transition<DTYPE>(out, m_numInputChannel, m_numOutputChannel, "Trans1");
        m_numInputChannel  = m_numOutputChannel;

        out                = this->MakeLayer(out, m_numOutputChannel, pBlockType, pNumOfBlock2, 1, "Block2");
        m_numInputChannel += pNumOfBlock2 * m_growthRate;
        m_numOutputChannel = (int)(floor(m_numInputChannel * m_reduction));
        out                = new Transition<DTYPE>(out, m_numInputChannel, m_numOutputChannel, "Trans2");
        m_numInputChannel  = m_numOutputChannel;

        out                = this->MakeLayer(out, m_numOutputChannel, pBlockType, pNumOfBlock3, 1, "Block3");
        m_numInputChannel += pNumOfBlock3 * m_growthRate;
        m_numOutputChannel = (int)(floor(m_numInputChannel * m_reduction));
        out                = new Transition<DTYPE>(out, m_numInputChannel, m_numOutputChannel, "Trans3");
        m_numInputChannel  = m_numOutputChannel;

        out                = this->MakeLayer(out, m_numOutputChannel, pBlockType, pNumOfBlock4, 1, "Block4");
        m_numInputChannel += pNumOfBlock4 * m_growthRate;
        m_numOutputChannel = (int)(floor(m_numInputChannel * m_reduction));
        out                = new Transition<DTYPE>(out, m_numInputChannel, m_numOutputChannel, "Trans4");
        m_numInputChannel  = m_numOutputChannel;

        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "BN1");
        out = new Relu<DTYPE>(out, "Relu1");

        out = new GlobalAvaragePooling2D<DTYPE>(out, "Avg Pooling");

        out = new ReShape<DTYPE>(out, 1, 1, m_numInputChannel, "ReShape");

        out = new Linear<DTYPE>(out, m_numInputChannel, pNumOfClass, TRUE, "Classification");
        out = new BatchNormalizeLayer<DTYPE>(out, FALSE, "BN0");

        this->AnalyzeGraph(out);

        // ======================= Select LossFunction Function ===================
        this->SetLossFunction(new SoftmaxCrossEntropy<float>(out, pLabel, "SCE"));
        // SetLossFunction(new MSE<float>(out, label, "MSE"));

        // ======================= Select Optimizer ===================
        // this->SetOptimizer(new GradientDescentOptimizer<float>(this->GetParameter(), 0.000001, 0.9, 5e-4, MINIMIZE));
        // this->SetOptimizer(new GradientDescentOptimizer<float>(this->GetParameter(), 0.001, MINIMIZE));
        this->SetOptimizer(new AdamOptimizer<float>(this->GetParameter(), 0.001, 0.9, 0.999, 1e-08, 5e-4, MINIMIZE));

        return TRUE;
    }

    Operator<DTYPE>* MakeLayer(Operator<DTYPE> *pInput, int pNumOfChannel, std::string pBlockType, int pNumOfBlock, int pStride, std::string pName = NULL) {
        if (pNumOfBlock == 0) {
            return pInput;
        } else if ((pBlockType == "DenseNetBlock") && (pNumOfBlock > 0)) {
            Operator<DTYPE> *out = pInput;

            for (int i = 0; i < pNumOfBlock; i++) {
                out            = new DenseNetBlock<DTYPE>(out, pNumOfChannel, m_growthRate, pName);
                pNumOfChannel += m_growthRate;
            }

            return out;
        } else if ((pBlockType == "Bottleneck") && (pNumOfBlock > 0)) {
            return NULL;
        } else return NULL;
    }
};

template<typename DTYPE> NeuralNetwork<DTYPE>* DenseNet18(Tensorholder<DTYPE> *pInput, Tensorholder<DTYPE> *pLabel, int pNumOfClass) {
    return new DenseNet<DTYPE>(pInput, pLabel, "DenseNetBlock", 2, 2, 2, 2, 12, 0.5, pNumOfClass);
}

#endif  // ifndef DENSENET_H_
