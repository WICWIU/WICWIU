#ifndef LSTM_H_
#define LSTM_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

#define TIME     0
#define BATCH    1

template<typename DTYPE> class LSTM : public Operator<DTYPE>{
private:

    //matmul, bias
    Operator<DTYPE> *m_aMatMulI2G;
    Operator<DTYPE> *m_aMatMulH2G;
    Operator<DTYPE> *m_aAddGates;
    Operator<DTYPE> *m_aAddBias;

    //forget Gate
    Operator<DTYPE> *m_aForgetGateInput;
    Operator<DTYPE> *m_aForgetGateSigmoid;

    //Input Gate
    Operator<DTYPE> *m_aInputGateInput;
    Operator<DTYPE> *m_aInputGateSigmoid;

    //Cell Gate
    Operator<DTYPE> *m_aCellGateInput;
    Operator<DTYPE> *m_aCellGateTanh;

    //Output Gate
    Operator<DTYPE> *m_aOutputGateInput;
    Operator<DTYPE> *m_aOutputGateSigmoid;

    //Cell state
    Operator<DTYPE> *m_aForgetGateCell;
    Operator<DTYPE> *m_aInputGateCell;
    Operator<DTYPE> *m_aCellState;

    //Hidden state
    Operator<DTYPE> *m_aBeforeHidden;
    Operator<DTYPE> *m_aHidden;

    //time
    Operator<DTYPE> *m_aTempHidden;
    Operator<DTYPE> *m_aTempCellState;

    //initHidden
    Operator<DTYPE> *m_pInitHidden;

#ifdef __CUDNN__

    //Tensor descriptor

    //cudnnTensorDescriptor_t x_desc[1], y_desc[1], dx_desc[1], dy_desc[1];
    cudnnTensorDescriptor_t *x_desc, *y_desc, *dx_desc, *dy_desc;


    cudnnTensorDescriptor_t hx_desc, dhx_desc, cx_desc, dcx_desc;

    cudnnTensorDescriptor_t hy_desc, dhy_desc, cy_desc, dcy_desc;

    //Filter descriptor
    cudnnFilterDescriptor_t w_desc, dw_desc;

    //dropout descriptor
    cudnnDropoutDescriptor_t dropout_desc;

    size_t state_size = 0;
    float m_droprate = 0.f;
    void *state = NULL;

    //RNN descriptor
    cudnnRNNDescriptor_t rnn_desc;
    cudnnRNNMode_t rnn_mode;
    cudnnRNNAlgo_t rnn_algo;


    //workspace
    void *workspace;
    void *reserved_space;

    //workspace size
    size_t workspace_size;
    size_t reserved_size;
    size_t weight_size;

    ///< cudnn 연산에서 사용 할 데이터를 가리키는 맴버 변수.
    DTYPE *x;     // input
    DTYPE *hx;    // input of initial hidden state
    DTYPE *cx;    // input of cell state (LSTM)

    DTYPE *y;     // output
    DTYPE *hy;    // output of final hidden state
    DTYPE *cy;    // output of final cell state (LSTM)

    DTYPE *dy;     // input of gradient
    DTYPE *dhy;    // input of final hidden state
    DTYPE *dcy;    // input of final cell state (LSTM)

    DTYPE *dx;     // output of gradient at the input of rnn
    DTYPE *dhx;    // output of gradient at the initial hidden state
    DTYPE *dcx;    // output of gradient at the initial cell state

    DTYPE *weights;
    DTYPE *gweights;

    //RNN dimensional information
    int dimA[3];
    int strideA[3];
    int dimW[3];


#endif  // __CUDNN__


public:
  LSTM(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIG, Operator<DTYPE> *pWeightHG, Operator<DTYPE> *lstmBias, Operator<DTYPE>* pInitHidden = NULL)
       : Operator<DTYPE>(4, pInput, pWeightIG, pWeightHG, lstmBias) {
      #if __DEBUG__
      std::cout << "LSTM::LSTM(Operator<DTYPE> *)" << '\n';
      #endif  // __DEBUG__
      this->Alloc(pInput, pWeightIG, pWeightHG, lstmBias, pInitHidden);
  }

    LSTM(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIG, Operator<DTYPE> *pWeightHG, Operator<DTYPE> *lstmBias, std::string pName, Operator<DTYPE>* pInitHidden = NULL)
         : Operator<DTYPE>(4, pInput, pWeightIG, pWeightHG, lstmBias) {
        #if __DEBUG__
        std::cout << "LSTM::LSTM(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIG, pWeightHG, lstmBias, pInitHidden);
    }

    ~LSTM() {
        #if __DEBUG__
        std::cout << "LSTM::~LSTM()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIG, Operator<DTYPE> *pWeightHG, Operator<DTYPE> *lstmBias, Operator<DTYPE>* pInitHidden) {


        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightIHShape = pWeightIG->GetResult()->GetShape();

        int hidTimeSize  = (*InputShape)[TIME];
        int hidBatchSize = (*InputShape)[BATCH];
        int hidColSize   = (*WeightIHShape)[3]/4;

        //time
        m_aTempHidden         = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempHidden");
        m_aTempCellState       = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempCell");

        //weight matmul, add bias
        m_aMatMulI2G            = new MatMul<DTYPE>(pWeightIG, pInput, "lstm_matmul_IG");
        m_aMatMulH2G            = new MatMul<DTYPE>(pWeightHG, m_aTempHidden, "lstm_matmul_HG");
        m_aAddGates              = new Addall<DTYPE>(m_aMatMulI2G, m_aMatMulH2G, "lstm_addall");
        m_aAddBias               = new AddColWise<DTYPE>(m_aAddGates, lstmBias, "lstm_F_addall");

        //forget Gate
        m_aForgetGateInput       = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "lstm_F_addall");
        m_aForgetGateSigmoid     = new Sigmoid<DTYPE>(m_aForgetGateInput, "lstm_f_sigmoid");

        //Input Gate
        m_aInputGateInput        = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "lstm_I_addall");
        m_aInputGateSigmoid      = new Sigmoid<DTYPE>(m_aInputGateInput, "lstm_I_sigmoid");

        //Cell Gate
        m_aCellGateInput         = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "lstm_c_addall");
        m_aCellGateTanh          = new Tanh<DTYPE>(m_aCellGateInput, "lstm_c_tanh");


        //Output Gate
        m_aOutputGateInput       = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "lstm_o_addall");
        m_aOutputGateSigmoid     = new Sigmoid<DTYPE>(m_aOutputGateInput, "lstm_o_sigmoid");

        //Cell state
        m_aForgetGateCell        = new Hadamard<DTYPE>(m_aTempCellState, m_aForgetGateSigmoid, "m_aForgetGateCell");
        m_aInputGateCell         = new Hadamard<DTYPE>(m_aCellGateTanh, m_aInputGateSigmoid, "beforecellstate");
        m_aCellState             = new Addall<DTYPE>(m_aForgetGateCell, m_aInputGateCell, "m_aCellState");

        //Hidden state
        m_aBeforeHidden          = new Tanh<DTYPE>(m_aCellState, "m_aBeforeHidden");
        m_aHidden                = new Hadamard<DTYPE>(m_aBeforeHidden, m_aOutputGateSigmoid, "cellstate");

        //for initHidden
        m_pInitHidden = pInitHidden;

        //For AnalyzeGraph
        pInput->GetOutputContainer()->Pop(m_aMatMulI2G);
        pWeightIG->GetOutputContainer()->Pop(m_aMatMulI2G);
        pWeightHG->GetOutputContainer()->Pop(m_aMatMulH2G);
        lstmBias->GetOutputContainer()->Pop(m_aAddBias);

        Shape *ResultShape = m_aHidden->GetResult()->GetShape();

        int timeSize  = (*ResultShape)[TIME];
        int batchSize = (*ResultShape)[BATCH];
        int colSize   = (*ResultShape)[4];

        this->SetResult(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));
        this->SetGradient(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));

        return TRUE;
    }

    #if __CUDNN__
          void InitializeAttributeForGPU(unsigned int idOfDevice) {

            Operator<DTYPE> *pInput    = this->GetInput()[0];
            Operator<DTYPE> *pWeightIH = this->GetInput()[1];

            Operator<DTYPE> *pWeightHG = this->GetInput()[2];

            Shape *InputShape    = pInput->GetResult()->GetShape();
            Shape *WeightXHShape = pWeightIH->GetResult()->GetShape();

            Shape *WeightHGShape = pWeightHG->GetResult()->GetShape();

            int colsizeOfInput = (*InputShape)[4];

            int hidTimeSize    = (*InputShape)[TIME];
            int hidBatchSize   = (*InputShape)[BATCH];
            int hidChannelSize = (*InputShape)[2];
            int hidColSize     = (*WeightHGShape)[3];


            if(m_pInitHidden != NULL)
              m_pInitHidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

            // std::cout<<"hidTimeSize : "<<hidTimeSize<<'\n';
            // std::cout<<"hidBatchSize : "<<hidBatchSize<<'\n';
            // std::cout<<"hidColSize : "<<hidColSize<<'\n';
            // std::cout<<"colsizeOfInput : "<<colsizeOfInput<<'\n';

            x_desc = new cudnnTensorDescriptor_t[hidTimeSize];
            dx_desc = new cudnnTensorDescriptor_t[hidTimeSize];
            y_desc = new cudnnTensorDescriptor_t[hidTimeSize];
            dy_desc = new cudnnTensorDescriptor_t[hidTimeSize];


            for(int i=0; i<hidTimeSize; i++){
                checkCUDNN(cudnnCreateTensorDescriptor(&x_desc[i]));
                checkCUDNN(cudnnCreateTensorDescriptor(&dx_desc[i]));

                checkCUDNN(cudnnCreateTensorDescriptor(&y_desc[i]));
                checkCUDNN(cudnnCreateTensorDescriptor(&dy_desc[i]));

                dimA[0] = hidBatchSize;
                dimA[1] = colsizeOfInput;
                dimA[2] = 1;

                strideA[0] = dimA[2] * dimA[1];
                strideA[1] = dimA[2];
                strideA[2] = 1;

                checkCUDNN(cudnnSetTensorNdDescriptor(x_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
                checkCUDNN(cudnnSetTensorNdDescriptor(dx_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));

                dimA[0] = hidBatchSize;
                dimA[1] = hidColSize;
                dimA[2] = 1;

                strideA[0] = dimA[2] * dimA[1];
                strideA[1] = dimA[2];
                strideA[2] = 1;

                checkCUDNN(cudnnSetTensorNdDescriptor(y_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
                checkCUDNN(cudnnSetTensorNdDescriptor(dy_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
            }


            dimA[0] = 1;
            dimA[1] = hidBatchSize;
            dimA[2] = hidColSize;

            strideA[0] = dimA[2] * dimA[1];
            strideA[1] = dimA[2];
            strideA[2] = 1;

            checkCUDNN(cudnnCreateTensorDescriptor(&hx_desc));      //initial hidden state
            checkCUDNN(cudnnCreateTensorDescriptor(&dhx_desc));
            checkCUDNN(cudnnCreateTensorDescriptor(&cx_desc));      //initial cell state
            checkCUDNN(cudnnCreateTensorDescriptor(&dcx_desc));

            checkCUDNN(cudnnCreateTensorDescriptor(&hy_desc));      //fianl hidden state
            checkCUDNN(cudnnCreateTensorDescriptor(&dhy_desc));
            checkCUDNN(cudnnCreateTensorDescriptor(&cy_desc));      //final cell state
            checkCUDNN(cudnnCreateTensorDescriptor(&dcy_desc));

            checkCUDNN(cudnnSetTensorNdDescriptor(hx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
            checkCUDNN(cudnnSetTensorNdDescriptor(cx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
            checkCUDNN(cudnnSetTensorNdDescriptor(dhx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
            checkCUDNN(cudnnSetTensorNdDescriptor(dcx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
            checkCUDNN(cudnnSetTensorNdDescriptor(hy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
            checkCUDNN(cudnnSetTensorNdDescriptor(cy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
            checkCUDNN(cudnnSetTensorNdDescriptor(dhy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
            checkCUDNN(cudnnSetTensorNdDescriptor(dcy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

            //dropout descriptor
            checkCUDNN(cudnnCreateDropoutDescriptor(&dropout_desc));
            checkCUDNN(cudnnDropoutGetStatesSize(this->GetCudnnHandle(), &state_size));
            checkCUDNN(cudnnSetDropoutDescriptor(dropout_desc, this->GetCudnnHandle(), m_droprate, state, state_size, time(NULL)));

            //rnn descriptor
            rnn_mode = CUDNN_LSTM;
            rnn_algo = CUDNN_RNN_ALGO_STANDARD;                                     // cudnnRNNAlgo_t

            checkCUDNN(cudnnCreateRNNDescriptor(&rnn_desc));
            checkCUDNN(cudnnSetRNNDescriptor_v6(this->GetCudnnHandle(), rnn_desc, hidColSize, 1, dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, rnn_mode, rnn_algo, CUDNN_DATA_FLOAT));

            //Bias
            //checkCUDNN(cudnnSetRNNBiasMode(rnn_desc, CUDNN_RNN_NO_BIAS));
            //checkCUDNN(cudnnSetRNNBiasMode(rnn_desc, CUDNN_RNN_SINGLE_REC_BIAS));
            checkCUDNN(cudnnSetRNNBiasMode(rnn_desc, CUDNN_RNN_SINGLE_INP_BIAS));

            //filter descriptor
            checkCUDNN(cudnnCreateFilterDescriptor(&w_desc));
            checkCUDNN(cudnnCreateFilterDescriptor(&dw_desc));

            checkCUDNN(cudnnGetRNNParamsSize(this->GetCudnnHandle(), rnn_desc, x_desc[0], &weight_size, CUDNN_DATA_FLOAT));

            dimW[0] = weight_size / sizeof(float);
            dimW[1] = 1;
            dimW[2] = 1;

            // std::cout<<"-------------------------------------------------------------------------"<<'\n';
            // std::cout<<"weight_size : "<<weight_size<<'\n';
            // std::cout<<"dimW[0] : "<<weight_size / sizeof(float)<<'\n';

            checkCUDNN(cudnnSetFilterNdDescriptor(w_desc,  CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));
            checkCUDNN(cudnnSetFilterNdDescriptor(dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));

            //workspace
            checkCUDNN(cudnnGetRNNWorkspaceSize(this->GetCudnnHandle(), rnn_desc,
                                                hidTimeSize,
                                                x_desc,
                                                &workspace_size));

            checkCUDNN(cudnnGetRNNTrainingReserveSize(this->GetCudnnHandle(), rnn_desc, hidTimeSize, x_desc, &reserved_size));

            if (workspace_size != 0) {
                checkCudaErrors(cudaMalloc(&workspace, workspace_size));

                if (workspace == NULL) {
                    printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                    exit(-1);
                }
            }

            if (reserved_size != 0) {
                checkCudaErrors(cudaMalloc(&reserved_space, reserved_size));

                if (reserved_space == NULL) {
                    printf("Failed to DEVICE allocation in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
                    exit(-1);
                }
            }


            int input_length = hidTimeSize * colsizeOfInput * hidBatchSize;
            int output_length = hidTimeSize * hidColSize * hidBatchSize;

           checkCudaErrors(cudaMalloc((DTYPE**)&x, input_length * sizeof(DTYPE)));
           checkCudaErrors(cudaMalloc((DTYPE**)&dx, input_length * sizeof(DTYPE)));
           checkCudaErrors(cudaMalloc((DTYPE**)&y, output_length * sizeof(DTYPE)));
           checkCudaErrors(cudaMalloc((DTYPE**)&dy, output_length * sizeof(DTYPE)));

           if(m_pInitHidden != NULL){
              checkCudaErrors(cudaMalloc((DTYPE**)&hx, hidColSize * hidBatchSize * sizeof(DTYPE)));
              checkCudaErrors(cudaMalloc((DTYPE**)&dhx, hidColSize * hidBatchSize * sizeof(DTYPE)));
            }

           checkCudaErrors(cudaMalloc((DTYPE**)&weights, weight_size));
           checkCudaErrors(cudaMalloc((DTYPE**)&gweights, weight_size));

          }

    #endif  // if __CUDNN__

    void Delete() {}

    int  ForwardPropagate(int pTime = 0) {

        if(pTime==0 && m_pInitHidden != NULL){

            Tensor<DTYPE> *initHidden = m_pInitHidden->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            int colsize      = initHidden->GetColSize();
            int batchsize    = initHidden->GetBatchSize();

            Shape *initShape = initHidden->GetShape();
            Shape *tempShape = tempHidden->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int co = 0; co < colsize; co++){
                    (*tempHidden)[Index5D(tempShape, pTime, ba, 0, 0, co)] = (*initHidden)[Index5D(initShape, pTime, ba, 0, 0, co)];
                }
            }
        }

        if(pTime != 0){

            //hidden
            Tensor<DTYPE> *prevHidden = m_aHidden->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            int batchsize      = prevHidden->GetBatchSize();
            int colSize        = prevHidden->GetColSize();
            Shape *HiddenShape = prevHidden->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int i = 0; i < colSize; i++) {
                    (*tempHidden)[Index5D(HiddenShape, pTime, ba, 0, 0, i)] = (*prevHidden)[Index5D(HiddenShape, pTime - 1, ba, 0, 0, i)];
                }
            }

            //Cell
            Tensor<DTYPE> *prevCellState = m_aCellState->GetResult();
            Tensor<DTYPE> *tempCellState = m_aTempCellState->GetResult();

            colSize        = prevCellState->GetColSize();
            Shape *CellShape = prevCellState->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int i = 0; i < colSize; i++) {
                    (*tempCellState)[Index5D(CellShape, pTime, ba, 0, 0, i)] = (*prevCellState)[Index5D(CellShape, pTime - 1, ba, 0, 0, i)];
                }
            }

        }

        m_aMatMulI2G->ForwardPropagate(pTime);
        m_aMatMulH2G->ForwardPropagate(pTime);
        m_aAddGates->ForwardPropagate(pTime);
        m_aAddBias->ForwardPropagate(pTime);

        Tensor<DTYPE> *tempForgetGates  = m_aForgetGateInput->GetResult();
        Tensor<DTYPE> *tempInputGates   = m_aInputGateInput->GetResult();
        Tensor<DTYPE> *tempCellGates    = m_aCellGateInput->GetResult();
        Tensor<DTYPE> *tempOutputGates  = m_aOutputGateInput->GetResult();

        Shape *EachShape = tempCellGates->GetShape();

        Tensor<DTYPE> *OneGates = m_aAddBias->GetResult();

        int batchsize      = OneGates->GetBatchSize();
        int h = m_aHidden->GetResult()->GetColSize();
        Shape *OneShape   = OneGates->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            for(int i=0; i<h; i++){
                (*tempForgetGates)[Index5D(EachShape, pTime, ba, 0, 0, i)]    = (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, i)];
                (*tempInputGates)[Index5D(EachShape, pTime, ba, 0, 0, i)]   = (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, h+i)];
                (*tempCellGates)[Index5D(EachShape, pTime, ba, 0, 0, i)]   = (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, 2*h+i)];
                (*tempOutputGates)[Index5D(EachShape, pTime, ba, 0, 0, i)] = (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, 3*h+i)];
            }
        }

        //Forget Gate
        m_aForgetGateInput->ForwardPropagate(pTime);
        m_aForgetGateSigmoid->ForwardPropagate(pTime);

        //Input Gate
        m_aInputGateInput->ForwardPropagate(pTime);
        m_aInputGateSigmoid->ForwardPropagate(pTime);

        //Cell Gate
        m_aCellGateInput->ForwardPropagate(pTime);
        m_aCellGateTanh->ForwardPropagate(pTime);

        //Output Gate
        m_aOutputGateInput->ForwardPropagate(pTime);
        m_aOutputGateSigmoid->ForwardPropagate(pTime);

        //Cell state
        m_aForgetGateCell->ForwardPropagate(pTime);
        m_aInputGateCell->ForwardPropagate(pTime);
        m_aCellState->ForwardPropagate(pTime);

        //Hidden state
        m_aBeforeHidden->ForwardPropagate(pTime);
        m_aHidden->ForwardPropagate(pTime);

        Tensor<DTYPE> *_result = m_aHidden->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        Shape *ResultShape = result->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            for (int i = 0; i < colSize; i++) {
                (*result)[Index5D(ResultShape, pTime, ba, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, ba, 0, 0, i)];
            }
        }

        return TRUE;
    }


    int BackPropagate(int pTime = 0) {

        Tensor<DTYPE> *_grad = m_aHidden->GetGradient();
        Tensor<DTYPE> *grad  = this->GetGradient();

        int batchsize        = grad->GetBatchSize();
        int colSize        = grad->GetColSize();
        int timeSize       = grad->GetTimeSize();
        Shape *ResultShape = grad->GetShape();


        for(int ba=0; ba<batchsize; ba++){
            for (int i = 0; i < colSize; i++) {
                (*_grad)[Index5D(ResultShape, pTime, ba, 0, 0, i)] = (*grad)[Index5D(ResultShape, pTime, ba, 0, 0, i)];
            }
        }

        if (pTime != timeSize-1) {

            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
            Tensor<DTYPE> *prevHiddenGrad = m_aHidden->GetGradient();

            int colSize        = tempHiddenGrad->GetColSize();
            Shape *HiddenShape = tempHiddenGrad->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int i = 0; i < colSize; i++) {
                    (*prevHiddenGrad)[Index5D(HiddenShape, pTime, ba, 0, 0, i)] += (*tempHiddenGrad)[Index5D(HiddenShape, pTime+1, ba, 0, 0, i)];
                }
            }
        }

        //Hidden state
        m_aHidden->BackPropagate(pTime);
        m_aBeforeHidden->BackPropagate(pTime);

        //cell
        if (pTime != timeSize-1) {

            Tensor<DTYPE> *tempCellGrad = m_aTempCellState->GetGradient();
            Tensor<DTYPE> *prevCellGrad = m_aCellState->GetGradient();

            int colSize        = tempCellGrad->GetColSize();
            Shape *CellShape = tempCellGrad->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int i = 0; i < colSize; i++) {
                    (*prevCellGrad)[Index5D(CellShape, pTime, ba, 0, 0, i)] += (*tempCellGrad)[Index5D(CellShape, pTime+1, ba, 0, 0, i)];
                }
            }
        }

        //Cell state
        m_aCellState->BackPropagate(pTime);
        m_aInputGateCell->BackPropagate(pTime);
        m_aForgetGateCell->BackPropagate(pTime);

        //Output Gate
        m_aOutputGateSigmoid->BackPropagate(pTime);
        m_aOutputGateInput->BackPropagate(pTime);

        //Cell gate
        m_aCellGateTanh->BackPropagate(pTime);
        m_aCellGateInput->BackPropagate(pTime);

        //Input Gates
        m_aInputGateSigmoid->BackPropagate(pTime);
        m_aInputGateInput->BackPropagate(pTime);

        //Forget Gates
        m_aForgetGateSigmoid->BackPropagate(pTime);
        m_aForgetGateInput->BackPropagate(pTime);

        //Gradient copy
        Tensor<DTYPE> *tempForgetGates  = m_aForgetGateInput->GetGradient();
        Tensor<DTYPE> *tempInputGates   = m_aInputGateInput->GetGradient();
        Tensor<DTYPE> *tempCellGates    = m_aCellGateInput->GetGradient();
        Tensor<DTYPE> *tempOutputGates  = m_aOutputGateInput->GetGradient();
        Shape *EachShape = tempCellGates->GetShape();

        Tensor<DTYPE> *OneGates = m_aAddBias->GetGradient();
        Shape *OneShape   = OneGates->GetShape();

        int h = m_aHidden->GetResult()->GetColSize();

        for(int ba=0; ba<batchsize; ba++){
            for(int i=0; i<h; i++){
                (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, i)]    = (*tempForgetGates)[Index5D(EachShape, pTime, ba, 0, 0, i)];
                (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, h+i)]   = (*tempInputGates)[Index5D(EachShape, pTime, ba, 0, 0, i)];
                (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, 2*h+i)]   = (*tempCellGates)[Index5D(EachShape, pTime, ba, 0, 0, i)];
                (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, 3*h+i)] = (*tempOutputGates)[Index5D(EachShape, pTime, ba, 0, 0, i)];
            }
        }

        m_aAddBias->BackPropagate(pTime);
        m_aAddGates->BackPropagate(pTime);
        m_aMatMulH2G->BackPropagate(pTime);
        m_aMatMulI2G->BackPropagate(pTime);

        if(pTime == 0 && m_pInitHidden != NULL){

            m_aMatMulH2G->BackPropagate(pTime);

            Tensor<DTYPE> *initHiddenGrad = m_pInitHidden->GetGradient();
            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();

            int colsize      = initHiddenGrad->GetColSize();
            int batchsize    = initHiddenGrad->GetBatchSize();
            Shape *initShape = initHiddenGrad->GetShape();
            Shape *tempShape = tempHiddenGrad->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for(int co=0; co<colsize; co++){
                    (*initHiddenGrad)[Index5D(initShape, 0, ba, 0, 0, co)] += (*tempHiddenGrad)[Index5D(tempShape, pTime, ba, 0, 0, co)];
                  }
            }
        }

        return TRUE;
    }

#if __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {

        Tensor<DTYPE> *input    = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *pWeightHG = this->GetInput()[2]->GetResult();
        Tensor<DTYPE> *result   = this->GetResult();

        weights = pWeightHG->GetGPUData(0);

        int timeSize        = input->GetTimeSize();

        // if(this->GetMode() == 0 && pTime != timeSize-1)
        //   return TRUE;

        cx = NULL;
        cy = NULL;
        hy = NULL;


        if(m_pInitHidden != NULL){

          Tensor<DTYPE> *initHidden = m_pInitHidden->GetResult();
          int m_Capacity = initHidden->GetCapacity();
          DTYPE * wicwiuInitHidden = initHidden->GetGPUData(0);

          checkCudaErrors(cudaMemcpy(&hx[0], wicwiuInitHidden, (m_Capacity * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
        }
        else{
          hx = NULL;
        }

        int m_CapacityPerTime = input->GetCapacity() / timeSize;

        for(int i=0; i<timeSize; i++){

            DTYPE *wicwiuX       = input->GetGPUData(i);
            checkCudaErrors(cudaMemcpy(&x[m_CapacityPerTime*i], wicwiuX, (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
        }

        checkCUDNN(cudnnRNNForwardTraining(this->GetCudnnHandle(), rnn_desc, timeSize,
                                           x_desc, x,
                                           hx_desc, hx,
                                           cx_desc, cx,
                                           w_desc, weights,
                                           y_desc, y,
                                           hy_desc, hy,
                                           cy_desc, cy,
                                           workspace, workspace_size,
                                           reserved_space, reserved_size));

        m_CapacityPerTime = result->GetCapacity() / timeSize;

        for(int i=0; i<timeSize; i++){
            DTYPE *wicwiuY       = result->GetGPUData(i);
            checkCudaErrors(cudaMemcpy(wicwiuY, &y[m_CapacityPerTime*i], (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));

        }
        return TRUE;
    }


    int BackPropagateOnGPU(int pTime = 0) {

        Tensor<DTYPE> *input             = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta       = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *pWeightHG          = this->GetInput()[2]->GetResult();
        Tensor<DTYPE> *pWeightHG_gradient = this->GetInput()[2]->GetGradient();
        Tensor<DTYPE> *result            = this->GetResult();
        Tensor<DTYPE> *this_delta        = this->GetDelta();


        weights     = pWeightHG->GetGPUData(0);
        gweights = pWeightHG_gradient->GetGPUData(0);

        int timeSize        = input->GetTimeSize();

        if(pTime != 0)
          return TRUE;

        dhy = NULL;
        dcy = NULL;
        cx = NULL;
        dcx = NULL;

        if(m_pInitHidden != NULL){

          Tensor<DTYPE> *initHidden = m_pInitHidden->GetResult();
          int m_Capacity = initHidden->GetCapacity();
          DTYPE * wicwiuInitHidden = initHidden->GetGPUData(0);

          checkCudaErrors(cudaMemcpy(&hx[0], wicwiuInitHidden, (m_Capacity * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
        }
        else
        {
            hx = NULL;
            dhx = NULL;
        }

        int m_CapacityPerTime = result->GetCapacity() / timeSize;

        for(int i=0; i<timeSize; i++){
            DTYPE *wicwiuY             = result->GetGPUData(i);
            DTYPE *wicwiuY_delta       = this_delta->GetGPUData(i);
            checkCudaErrors(cudaMemcpy(&y[m_CapacityPerTime*i], wicwiuY, (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(&dy[m_CapacityPerTime*i], wicwiuY_delta, (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));

        }

        checkCUDNN(cudnnRNNBackwardData(this->GetCudnnHandle(), rnn_desc, timeSize,
                                        y_desc, y,
                                        dy_desc, dy,
                                        dhy_desc, dhy,
                                        dcy_desc, dcy,
                                        w_desc, weights,
                                        hx_desc, hx,
                                        cx_desc, cx,
                                        dx_desc, dx,
                                        dhx_desc, dhx,
                                        dcx_desc, dcx,
                                        workspace, workspace_size,
                                        reserved_space, reserved_size));

        m_CapacityPerTime = input->GetCapacity() / timeSize;

        for(int i=0; i<timeSize; i++){
            DTYPE *wicwiuX_delta       = input_delta->GetGPUData(i);
            checkCudaErrors(cudaMemcpy(wicwiuX_delta, &dx[m_CapacityPerTime*i], (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));

        }

        if(m_pInitHidden != NULL){

          Tensor<DTYPE> *initHidden = m_pInitHidden->GetGradient();
          int m_Capacity = initHidden->GetCapacity();
          DTYPE * wicwiuInitHidden = initHidden->GetGPUData(0);

          checkCudaErrors(cudaMemcpy(wicwiuInitHidden, &dhx[0], (m_Capacity * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
        }

        m_CapacityPerTime = input->GetCapacity() / timeSize;

        for(int i=0; i<timeSize; i++){

            DTYPE *wicwiuX       = input->GetGPUData(i);
            checkCudaErrors(cudaMemcpy(&x[m_CapacityPerTime*i], wicwiuX, (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
        }

        checkCUDNN(cudnnRNNBackwardWeights(this->GetCudnnHandle(), rnn_desc, timeSize,
                                           x_desc, x,
                                           hx_desc, hx,
                                           y_desc, y,
                                           workspace, workspace_size,
                                           dw_desc, gweights,
                                           reserved_space, reserved_size));

        return TRUE;
    }
#endif  // if __CUDNN__


    int ResetResult() {
        //time
        m_aTempHidden->ResetResult();
        m_aTempCellState->ResetResult();

        //weight matmul, add bias
        m_aMatMulI2G->ResetResult();
        m_aMatMulH2G->ResetResult();
        m_aAddGates->ResetResult();
        m_aAddBias->ResetResult();

        //forget Gate
        m_aForgetGateInput->ResetResult();
        m_aForgetGateSigmoid->ResetResult();

        //Input Gate
        m_aInputGateInput->ResetResult();
        m_aInputGateSigmoid->ResetResult();

        //???
        m_aCellGateInput->ResetResult();
        m_aCellGateTanh->ResetResult();

        //Output Gate
        m_aOutputGateInput->ResetResult();
        m_aOutputGateSigmoid->ResetResult();

        //Cell state
        m_aForgetGateCell->ResetResult();
        m_aInputGateCell->ResetResult();
        m_aCellState->ResetResult();

        //Hidden state
        m_aBeforeHidden->ResetResult();
        m_aHidden->ResetResult();

        //initHidden
        if(m_pInitHidden != NULL)
           m_pInitHidden->ResetResult();

        Tensor<DTYPE> *result = this->GetResult();
        result->Reset();

        return TRUE;
    }

    int ResetGradient() {
        //time
        m_aTempHidden->ResetGradient();
        m_aTempCellState->ResetGradient();

        //weight matmul, add bias
        m_aMatMulI2G->ResetGradient();
        m_aMatMulH2G->ResetGradient();
        m_aAddGates->ResetGradient();
        m_aAddBias->ResetGradient();

        //forget Gate
        m_aForgetGateInput->ResetGradient();
        m_aForgetGateSigmoid->ResetGradient();

        //Input Gate
        m_aInputGateInput->ResetGradient();
        m_aInputGateSigmoid->ResetGradient();

        m_aCellGateInput->ResetGradient();
        m_aCellGateTanh->ResetGradient();

        //Output Gate
        m_aOutputGateInput->ResetGradient();
        m_aOutputGateSigmoid->ResetGradient();

        //Cell state
        m_aForgetGateCell->ResetGradient();
        m_aInputGateCell->ResetGradient();
        m_aCellState->ResetGradient();

        //Hidden state
        m_aBeforeHidden->ResetGradient();
        m_aHidden->ResetGradient();

        //initHidden
        if(m_pInitHidden != NULL)
           m_pInitHidden->ResetGradient();

        Tensor<DTYPE> *grad = this->GetGradient();
        grad->Reset();

        return TRUE;
    }

};


#endif  // LSTM_H_
