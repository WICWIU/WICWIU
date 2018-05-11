#ifndef DENSEBLOCK_H_
#define DENSEBLOCK_H_    value

#include "..//Operator.h"

template<typename DTYPE> class DenseBlock : public Operator<DTYPE>{
private:
    Operator<DTYPE> *m_pBlockOutput;
    Operator<DTYPE> **m_apInput;
    Operator<DTYPE> **m_apBN;
    Operator<DTYPE> **m_apBN_Relu;
    Operator<DTYPE> **m_apConv;
    Operator<DTYPE> **m_apConv_Relu;

    int m_nChannel;
    int m_nBlockLayer;

public:
    DenseBlock(NeuralNetwork<float>& nn, Operator<DTYPE> *pInput, int growthRate, int nBlockLayer, std::string pName) : Operator<DTYPE>(pName) {
        std::cout << "DenseBlock::DenseBlock(Operator<DTYPE> *, int, int, std::string)" << '\n';
        this->Alloc(nn, pInput, growthRate, nBlockLayer);
    }

    ~DenseBlock() {
        std::cout << "DenseBlock::~DenseBlock()" << '\n';
        Delete();
    }

    void Delete() {
        if (m_apInput) {
            delete[] m_apInput;
            m_apInput = NULL;
        }

        if (m_apBN) {
            delete[] m_apBN;
            m_apBN = NULL;
        }

        if (m_apBN_Relu) {
            delete[] m_apBN_Relu;
            m_apBN_Relu = NULL;
        }

        if (m_apConv) {
            delete[] m_apConv;
            m_apConv = NULL;
        }

        if (m_apConv_Relu) {
            delete[] m_apConv_Relu;
            m_apConv_Relu = NULL;
        }
    }

    int Alloc(NeuralNetwork<float>& nn, Operator<DTYPE> *pInput, int growthRate, int nBlockLayer) {
        std::cout << "DenseBlock::Alloc(Operator<DTYPE> *, int, int)" << '\n';
        m_pBlockOutput = NULL;
        m_apInput      = NULL;
        m_apBN         = NULL;
        m_apBN_Relu    = NULL;
        m_apConv       = NULL;
        m_apConv_Relu  = NULL;
        m_nChannel     = 0;
        m_nBlockLayer  = nBlockLayer;

        MakeDenseBlock(nn, pInput, growthRate, nBlockLayer);

        if (m_pBlockOutput == NULL) {
            printf("Failed to make DenseBlock in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }

        Shape *shapeOfResult = new Shape(m_pBlockOutput->GetResult()->GetShape());
        int    t = (*shapeOfResult)[0]; int b = (*shapeOfResult)[1];
        int    r = (*shapeOfResult)[3]; int c = (*shapeOfResult)[4];

        this->SetResult(new Tensor<DTYPE>(t, b, m_nChannel, r, c));
        this->SetDelta(new Tensor<DTYPE>(t, b, m_nChannel, r, c));


        // Operator<DTYPE>::Alloc(1, m_pBlockOutput);

        return TRUE;
    }

    void MakeDenseBlock(NeuralNetwork<float>& nn, Operator<DTYPE> *pInput, int growthRate, int nBlockLayer) {
        Operator<DTYPE> *op = pInput;
        m_apInput     = new Operator<DTYPE> *[nBlockLayer + 1];
        m_apBN        = new Operator<DTYPE> *[nBlockLayer];
        m_apBN_Relu   = new Operator<DTYPE> *[nBlockLayer];
        m_apConv      = new Operator<DTYPE> *[nBlockLayer];
        m_apConv_Relu = new Operator<DTYPE> *[nBlockLayer];


        m_nChannel = pInput->GetResult()->GetChannelSize();

        for (int i = 0; i < nBlockLayer; i++) {
            m_apInput[i] = op;
            // op = MakeDenseLayer(nn, op, growthRate, m_nChannel, i);
            op          = MakeDenseLayer(nn, m_apInput[i], growthRate, m_nChannel, i);
            m_nChannel += growthRate;
            op          = this->Concatenate(m_apInput[i], op, 2);
        }
        m_pBlockOutput = op;
    }

    // BN - RELU - CONV(3x3)
    // growthRate : each layer adds feature-maps of its own to this state. ('k' in paper)
    Operator<DTYPE>* MakeDenseLayer(NeuralNetwork<float>& nn, Operator<DTYPE> *pInput, int growthRate, int inputChannel, int layerNum) {
        Operator<DTYPE> *op        = NULL;
        Tensorholder<float> *w     = new Tensorholder<float>(Tensor<float>::Truncated_normal(1, growthRate, inputChannel, 3, 3, 0.0, 0.1), "weight");
        Tensorholder<float> *gamma = new Tensorholder<float>(Tensor<float>::Constants(1, 1, growthRate, 1, 1, 1), "scale");
        Tensorholder<float> *beta  = new Tensorholder<float>(Tensor<float>::Zeros(1, 1, growthRate, 1, 1), "shift");

        m_apBN[layerNum]        = new BatchNormalize<float>(pInput, gamma, beta, 60000 / 10, 10000 / 10, TRUE, "DenseBlock_batchnormalize");
        m_apBN_Relu[layerNum]   = new Relu<float>(m_apBN[layerNum], "DenseBlock_bn_relu");
        m_apConv[layerNum]      = new Convolution2D<float>(m_apBN_Relu[layerNum], w, 1, 1, 1, 1, SAME, "DenseBlock_convolution");
        m_apConv_Relu[layerNum] = new Relu<float>(m_apConv[layerNum], "DenseBlock_conv_relu");

        // op = nn.AddOperator(new BatchNormalize<float>(pInput, gamma, beta, 60000 / 10, 10000 / 10, TRUE, "DenseBlock_batchnormalize"));
        // op = nn.AddOperator(new Relu<float>(op, "DenseBlock_bn_relu"));
        // op = nn.AddOperator(new Convolution2D<float>(op, w, 1, 1, 1, 1, SAME, "DenseBlock_convolution"));
        // op = nn.AddOperator(new Relu<float>(op, "DenseBlock_conv_relu"));

        return m_apConv_Relu[layerNum];
    }

    int ForwardPropagate() {
        Operator<DTYPE> *concat = NULL;

        for (int i = 0; i < m_nBlockLayer - 1; i++) {
            DenseLayerForwardPropagate(i);
            concat = this->Concatenate(m_apInput[i], m_apConv_Relu[i], 2);

            Tensor<DTYPE> *input        = concat->GetResult();
            Tensor<DTYPE> *input_delta  = concat->GetDelta();
            Tensor<DTYPE> *output       = m_apInput[i + 1]->GetResult();
            Tensor<DTYPE> *output_delta = m_apInput[i + 1]->GetDelta();
            int outputCapacity          = output->GetCapacity();

            for (int i = 0; i < outputCapacity; i++) {
                (*output)[i]       = (*input)[i];
                (*output_delta)[i] = (*input_delta)[i];
            }
        }

        int lastLayerNum = m_nBlockLayer - 1;
        DenseLayerForwardPropagate(lastLayerNum);
        concat = this->Concatenate(m_apInput[lastLayerNum], m_apConv_Relu[lastLayerNum], 2);

        Tensor<DTYPE> *input       = concat->GetResult();
        Tensor<DTYPE> *input_delta = concat->GetDelta();

        Tensor<DTYPE> *output       = this->GetResult();
        Tensor<DTYPE> *output_delta = this->GetDelta();
        int outputCapacity          = output->GetCapacity();

        for (int i = 0; i < outputCapacity; i++) {
            (*output)[i]       = (*input)[i];
            (*output_delta)[i] = (*input_delta)[i];
        }


        return TRUE;
    }

    void DenseLayerForwardPropagate(int layerNum) {
        m_apBN[layerNum]->ForwardPropagate();
        m_apBN_Relu[layerNum]->ForwardPropagate();
        m_apConv[layerNum]->ForwardPropagate();
        m_apConv_Relu[layerNum]->ForwardPropagate();
    }

    // concatCapacity : input capacity before concatenation.
    int BackPropagate() {
        int lastLayerNum   = m_nBlockLayer - 1;
        int outputCapacity = this->GetResult()->GetCapacity();
        int concatCapacity = m_apInput[lastLayerNum]->GetResult()->GetCapacity();
        int k              = outputCapacity - concatCapacity;

        // k-feature maps
        // Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *input_delta = m_apConv_Relu[lastLayerNum]->GetDelta();
        Tensor<DTYPE> *this_delta  = this->GetDelta();

        for (int i = 0; i < k; i++) {
            (*input_delta)[i] = (*this_delta)[i + concatCapacity];
        }

        DenseLayerBackPropagate(lastLayerNum);

        Tensor<DTYPE> *concat_delta = m_apBN[lastLayerNum]->GetInput()[0]->GetDelta();

        for (int i = 0; i < concatCapacity; i++) {
            (*concat_delta)[i] += (*this_delta)[i];
        }

        for (int i = lastLayerNum - 1; i > 0; i--) {
            int outputCapacity = m_apConv_Relu[i]->GetResult()->GetCapacity();
            int concatCapacity = m_apInput[i]->GetResult()->GetCapacity();
            int k              = outputCapacity - concatCapacity;

            // k-feature maps
            Tensor<DTYPE> *input_delta = m_apConv_Relu[i]->GetInput()[0]->GetDelta();
            Tensor<DTYPE> *this_delta  = m_apConv_Relu[i]->GetDelta();

            for (int i = 0; i < k; i++) {
                (*input_delta)[i] = (*this_delta)[concatCapacity + i];
            }

            DenseLayerBackPropagate(i);

            Tensor<DTYPE> *concat_delta = m_apBN[i]->GetInput()[0]->GetDelta();

            for (int i = 0; i < concatCapacity; i++) {
                (*concat_delta)[i] += (*this_delta)[i];
            }
        }
        return TRUE;
    }

    void DenseLayerBackPropagate(int layerNum) {
        m_apConv_Relu[layerNum]->BackPropagate();
        m_apConv[layerNum]->BackPropagate();
        m_apBN_Relu[layerNum]->BackPropagate();
        m_apBN[layerNum]->BackPropagate();
    }
};

#endif  // DENSEBLOCK_H_
