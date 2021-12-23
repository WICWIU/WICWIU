#ifndef GRU_H_
#define GRU_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

#define TIMEIDX     0
#define BATCHIDX    1

/*!
@class GRU GRU class
*/

template<typename DTYPE> class GRU : public Operator<DTYPE>{
private:

    //matmul, bias
    Operator<DTYPE> *m_aMatMulI2G;
    Operator<DTYPE> *m_aMatMulH2RZ;
    Operator<DTYPE> *m_aAddGates;
    Operator<DTYPE> *m_aAddgBias;

    //Reset Gate
    Operator<DTYPE> *m_aRGateInput;
    Operator<DTYPE> *m_aRGateSigmoid;

    //Update Gate(z)
    Operator<DTYPE> *m_aZGateInput;
    Operator<DTYPE> *m_aZGateSigmoid;

    //Candidate Hidden
    Operator<DTYPE> *m_aMatMulI2CH;     //matmul
    Operator<DTYPE> *m_aRAndHidden;      //hadamard
    Operator<DTYPE> *m_aMatMulH2CH;     //matmul
    Operator<DTYPE> *m_aBeforeCandidateHiddenInput;
    Operator<DTYPE> *m_aCandidateHiddenInput;            //bias
    Operator<DTYPE> *m_aCandidateHiddenTanh;

    //Hidden state
    Operator<DTYPE> *m_aBeforeZHidden;
    Operator<DTYPE> *m_aBeforeGHidden1;
    Operator<DTYPE> *m_aBeforeGHidden2;
    Operator<DTYPE> *m_aHidden;

    //Onettme
    Operator<DTYPE> *m_aOneTensor;

    //time
    Operator<DTYPE> *m_aTempHidden;

    //init hidden
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
    DTYPE *x;                // input
    DTYPE *hx;    // input of initial hidden state
    DTYPE *cx;    // input of cell state (LSTM)

    DTYPE *y;                // output
    DTYPE *hy;    // output of final hidden state
    DTYPE *cy;    // output of final cell state (LSTM)

    DTYPE *dy;               // input of gradient
    DTYPE *dhy;    // input of final hidden state
    DTYPE *dcy;    // input of final cell state (LSTM)

    DTYPE *dx;               // output of gradient at the input of rnn
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
    /**
    * @brief GRU의 생성자
    * @details 파라미터로 받은 pInput, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias, pInitHidden로 Alloc한다.
    * @param pInput GRU의 input Operator
    * @param pWeightIG Input to Gate에 해당하는 matmtul의 weight
    * @param pWeightHG hidden to gate에 해당하는 matmul의 weight
    * @param pWeightICH input to Candidate hidden에 해당하는 matmul의 weight
    * @param pWeightHCH hidden to Candidate hidden에 해당하는 matmul의 weight
    * @param gBias gate의 bias
    * @param chBias Candidate hidden의 bias
    * @param pInitHidden  GRU의 초기 hidden 값
    */
    GRU(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIG, Operator<DTYPE> *pWeightHG, Operator<DTYPE> *pWeightICH, Operator<DTYPE> *pWeightHCH, Operator<DTYPE> *gBias, Operator<DTYPE> *chBias, Operator<DTYPE>* pInitHidden = NULL)
        : Operator<DTYPE>(7, pInput, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias) {
        #if __DEBUG__
        std::cout << "GRU::GRU(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias, pInitHidden);
    }

    /**
     * @brief Construct a new GRU object
     * @details 파라미터로 받은 pInput, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias, pInitHidden로 Alloc한다.
     * @param pInput GRU의 input Operator
     * @param pWeightIG Input to Gate에 해당하는 matmtul의 weight
     * @param pWeightHG hidden to gate에 해당하는 matmul의 weight
     * @param pWeightICH input to Candidate hidden에 해당하는 matmul의 weight
     * @param pWeightHCH hidden to Candidate hidden에 해당하는 matmul의 weight
     * @param gBias gate의 bias
     * @param chBias Candidate hidden의 bias
     * @param pName  사용자가 부여한 Operator 이름
     * @param pInitHidden GRU의 초기 hidden 값
     */
    GRU(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIG, Operator<DTYPE> *pWeightHG, Operator<DTYPE> *pWeightICH, Operator<DTYPE> *pWeightHCH, Operator<DTYPE> *gBias, Operator<DTYPE> *chBias, std::string pName, Operator<DTYPE>* pInitHidden = NULL)
         : Operator<DTYPE>(7, pInput, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias, pInitHidden) {
        #if __DEBUG__
        std::cout << "GRU::GRU(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias, pInitHidden);
    }

    ~GRU() {
        #if __DEBUG__
        std::cout << "GRU::~GRU()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    /**
     * @brief 파라미터로 받은  pInput, pWeightIG, pWeightHG, pWeightICH, pWeightHCH, gBias, chBias, pInitHidden으로 맴버 변수들을 초기화 한다.
     * @details GRU연산에 필요한 MatMul, add, Sigmod, Minus, Tanh, Hadamard operator를 맴버 변수로 생성한다.
     * @param pInput GRU의 input Operator
     * @param pWeightIG Input to Gate에 해당하는 matmtul의 weight
     * @param pWeightHG hidden to gate에 해당하는 matmul의 weight
     * @param pWeightICH input to Candidate hidden에 해당하는 matmul의 weight
     * @param pWeightHCH hidden to Candidate hidden에 해당하는 matmul의 weight
     * @param gBias gate의 bias
     * @param chBias Candidate hidden의 bias
     * @param pInitHidden  GRU의 초기 hidden 값
     * @return int 
     */
    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIG, Operator<DTYPE> *pWeightHG, Operator<DTYPE> *pWeightICH, Operator<DTYPE> *pWeightHCH, Operator<DTYPE> *gBias, Operator<DTYPE> *chBias, Operator<DTYPE>* pInitHidden) {

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightIGShape = pWeightIG->GetResult()->GetShape();

        int hidTimeSize  = (*InputShape)[TIMEIDX];
        int hidBatchSize = (*InputShape)[BATCHIDX];
        int hidColSize   = (*WeightIGShape)[3]/2;

        //Onetensor
        m_aOneTensor    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(hidTimeSize, hidBatchSize, 1, 1, hidColSize, 1.0), "tempHidden");

        //time
        m_aTempHidden   = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempHidden");

        //Reset&update gate matmul, bias
        m_aMatMulI2G    = new MatMul<DTYPE>(pWeightIG, pInput, "gru_matmul_IG");
        m_aMatMulH2RZ   = new MatMul<DTYPE>(pWeightHG, m_aTempHidden, "gru_matmul_IG");
        m_aAddGates      = new Addall<DTYPE>(m_aMatMulI2G, m_aMatMulH2RZ, "gru_addall");
        m_aAddgBias      = new AddColWise<DTYPE>(m_aAddGates, gBias, "gru_F_addall");

        //R Gate
        m_aRGateInput    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "gru_R_addall");
        m_aRGateSigmoid  = new Sigmoid<DTYPE>(m_aRGateInput, "gru_R_sigmoid");

        //Z Gate
        m_aZGateInput    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "gru_Z_addall");
        m_aZGateSigmoid  = new Sigmoid<DTYPE>(m_aZGateInput, "gru_Z_sigmoid");

        //Candidate Hidden
        m_aMatMulI2CH   = new MatMul<DTYPE>(pWeightICH, pInput, "gru_matmul_IG");
        m_aRAndHidden    = new Hadamard<DTYPE>(m_aRGateSigmoid, m_aTempHidden, "ForgetGateCell");
        m_aMatMulH2CH   = new MatMul<DTYPE>(pWeightHCH, m_aRAndHidden, "gru_matmul_IG");
        m_aBeforeCandidateHiddenInput  = new Addall<DTYPE>(m_aMatMulI2CH, m_aMatMulH2CH, "gru_addall");
        m_aCandidateHiddenInput        = new AddColWise<DTYPE>(m_aBeforeCandidateHiddenInput, chBias, "gru_F_addall");            //bias
        m_aCandidateHiddenTanh         = new Tanh<DTYPE>(m_aCandidateHiddenInput, "lstm_c_tanh");

        //Hidden state
        m_aBeforeZHidden  = new Hadamard<DTYPE>(m_aZGateSigmoid, m_aTempHidden, "ForgetGateCell");
        m_aBeforeGHidden1 = new Minus<DTYPE>(m_aOneTensor, m_aZGateSigmoid, "new data");
        m_aBeforeGHidden2 = new Hadamard<DTYPE>(m_aBeforeGHidden1, m_aCandidateHiddenTanh, "ForgetGateCell");
        m_aHidden         = new Addall<DTYPE>(m_aBeforeZHidden, m_aBeforeGHidden2, "gru_addall");

        //for initHidden
        m_pInitHidden = pInitHidden;

        //For AnalyzeGraph
        pInput->GetOutputContainer()->Pop(m_aMatMulI2G);
        pInput->GetOutputContainer()->Pop(m_aMatMulI2CH);
        pWeightIG->GetOutputContainer()->Pop(m_aMatMulI2G);
        pWeightHG->GetOutputContainer()->Pop(m_aMatMulH2RZ);
        pWeightICH->GetOutputContainer()->Pop(m_aMatMulI2CH);
        pWeightHCH->GetOutputContainer()->Pop(m_aMatMulH2CH);
        gBias->GetOutputContainer()->Pop(m_aAddgBias);
        chBias->GetOutputContainer()->Pop(m_aCandidateHiddenInput);

        Shape *ResultShape = m_aHidden->GetResult()->GetShape();

        int timeSize  = (*ResultShape)[TIMEIDX];
        int batchSize = (*ResultShape)[BATCHIDX];
        int colSize   = (*ResultShape)[4];

        this->SetResult(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));
        this->SetGradient(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));

        return TRUE;
    }


#if __CUDNN__
        /**
        * @brief cudnn을 사용하기 전 관련 맴버변수들을 초기화 한다.
        * @details TensorDesriptor들을 생성하고, TensorDesriptor들의 데이터가 batch, channel, row, col 순서로 배치되도록 지정한다.
        * @details GRU연산에 필요한 알고리즘을 정의하고, 연산에 필요한 메모리공간을 할당 받는다.
        * @param idOfDevice 사용할 GPU의 id
        */
      void InitializeAttributeForGPU(unsigned int idOfDevice) {

        Operator<DTYPE> *pInput    = this->GetInput()[0];
        Operator<DTYPE> *pWeightHG = this->GetInput()[2];

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightHGShape = pWeightHG->GetResult()->GetShape();

        int colsizeOfInput = (*InputShape)[4];

        int hidTimeSize    = (*InputShape)[TIMEIDX];
        int hidBatchSize   = (*InputShape)[BATCHIDX];
        int hidChannelSize = (*InputShape)[2];
        int hidColSize     = (*WeightHGShape)[3];

        // std::cout<<"hidTimeSize : "<<hidTimeSize<<'\n';
        // std::cout<<"hidBatchSize : "<<hidBatchSize<<'\n';
        // std::cout<<"hidColSize : "<<hidColSize<<'\n';
        // std::cout<<"colsizeOfInput : "<<colsizeOfInput<<'\n';

        if(m_pInitHidden != NULL)
            m_pInitHidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);

        x_desc = new cudnnTensorDescriptor_t[hidTimeSize];
        dx_desc = new cudnnTensorDescriptor_t[hidTimeSize];
        y_desc = new cudnnTensorDescriptor_t[hidTimeSize];
        dy_desc = new cudnnTensorDescriptor_t[hidTimeSize];


//tensor descriptor
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

        //tensor descriptor
        checkCUDNN(cudnnCreateTensorDescriptor(&hx_desc));      //initial hidden state
        checkCUDNN(cudnnCreateTensorDescriptor(&dhx_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&cx_desc));      //initial cell state
        checkCUDNN(cudnnCreateTensorDescriptor(&dcx_desc));

        checkCUDNN(cudnnCreateTensorDescriptor(&hy_desc));      //fianl hidden state
        checkCUDNN(cudnnCreateTensorDescriptor(&dhy_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&cy_desc));      //final cell state
        checkCUDNN(cudnnCreateTensorDescriptor(&dcy_desc));

        //tensor descriptor
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
        rnn_mode = CUDNN_GRU;
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

//workspace    hidTimeSize
        checkCUDNN(cudnnGetRNNWorkspaceSize(this->GetCudnnHandle(), rnn_desc,
                                            hidTimeSize,
                                            x_desc,
                                            &workspace_size));

        checkCUDNN(cudnnGetRNNTrainingReserveSize(this->GetCudnnHandle(), rnn_desc, hidTimeSize, x_desc, &reserved_size));

  //      checkCudaErrors(cudaMalloc(&workspace, workspace_size));
  //      checkCudaErrors(cudaMalloc(&reserved_space, reserved_size));

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

    /**
     * @brief GRU의 ForwardPropagate 메소드
     * @details GRU 연산 중 Input to gate & candidate hidden, hidden to gate & candidate hidden, activation function인 tanh 연산을 수행한다.
     * @details tempHidden이라는 Operator를 사용하여 t-1 에서 t로 가는 hidden 2 hidden 연산을 수행한다.
     * @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
     * @return int 
     */
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
        }

        m_aMatMulI2G->ForwardPropagate(pTime);
        m_aMatMulH2RZ->ForwardPropagate(pTime);
        m_aAddGates->ForwardPropagate(pTime);
        m_aAddgBias->ForwardPropagate(pTime);

        Tensor<DTYPE> *tempRGates  = m_aRGateInput->GetResult();
        Tensor<DTYPE> *tempZGates   = m_aZGateInput->GetResult();


        Shape *EachShape = tempRGates->GetShape();
        Tensor<DTYPE> *OneGates = m_aAddgBias->GetResult();

        int batchsize = OneGates->GetBatchSize();
        int h = m_aHidden->GetResult()->GetColSize();
        Shape *OneShape   = OneGates->GetShape();

        for(int ba=0; ba<batchsize; ba++){
            for(int i=0; i<h; i++){
                (*tempRGates)[Index5D(EachShape, pTime, ba, 0, 0, i)]   = (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, i)];
                (*tempZGates)[Index5D(EachShape, pTime, ba, 0, 0, i)]   = (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, h+i)];
            }
        }

        //R Gate
        m_aRGateSigmoid->ForwardPropagate(pTime);

        //Z Gate
        m_aZGateSigmoid->ForwardPropagate(pTime);

        //Candidate Hidden
        m_aMatMulI2CH->ForwardPropagate(pTime);
        m_aRAndHidden->ForwardPropagate(pTime);
        m_aMatMulH2CH->ForwardPropagate(pTime);
        m_aBeforeCandidateHiddenInput->ForwardPropagate(pTime);
        m_aCandidateHiddenInput->ForwardPropagate(pTime);
        m_aCandidateHiddenTanh->ForwardPropagate(pTime);

        //Hidden state
        m_aBeforeZHidden->ForwardPropagate(pTime);
        m_aBeforeGHidden1->ForwardPropagate(pTime);
        m_aBeforeGHidden2->ForwardPropagate(pTime);
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


    /**
     * @brief GRU의 BackPropagate 메소드
     * @details GRU 연산의 미분값을 계산하여 input_delta, WeightIG_delta, WeightHG_delta, pWeightICH_delta, WeightHCH_delta, gbias_delta, chBias_delta, InitHidden_delta에 각각 더해 넣는다.
     * @details tempHidden이라는 Operator를 사용하여 t+1 에서 t로 가는 hidden 2 hidden에 해당하는 Gradient 전달하여준다.
     * @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
     * @return int 
     */
    int BackPropagate(int pTime = 0) {

        Tensor<DTYPE> *_grad = m_aHidden->GetGradient();
        Tensor<DTYPE> *grad  = this->GetGradient();

        int colSize        = grad->GetColSize();
        int timeSize       = grad->GetTimeSize();
        int batchsize      = grad->GetBatchSize();
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
        m_aBeforeGHidden2->BackPropagate(pTime);
        m_aBeforeGHidden1->BackPropagate(pTime);
        m_aBeforeZHidden->BackPropagate(pTime);

        //Candidate Hidden
        m_aCandidateHiddenTanh->BackPropagate(pTime);
        m_aCandidateHiddenInput->BackPropagate(pTime);
        m_aBeforeCandidateHiddenInput->BackPropagate(pTime);
        m_aMatMulH2CH->BackPropagate(pTime);
        m_aRAndHidden->BackPropagate(pTime);
        m_aMatMulI2CH->BackPropagate(pTime);

        //Z Gate
        m_aZGateSigmoid->BackPropagate(pTime);

        //R Gate
        m_aRGateSigmoid->BackPropagate(pTime);

        Tensor<DTYPE> *tempRGates  = m_aRGateInput->GetGradient();
        Tensor<DTYPE> *tempZGates   = m_aZGateInput->GetGradient();

        Shape *EachShape = tempZGates->GetShape();

        Tensor<DTYPE> *OneGates = m_aAddgBias->GetGradient();
        Shape *OneShape   = OneGates->GetShape();

        int h = m_aHidden->GetResult()->GetColSize();

        for(int ba=0; ba<batchsize; ba++){
            for(int i=0; i<h; i++){
                (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, i)]    = (*tempRGates)[Index5D(EachShape, pTime, ba, 0, 0, i)];
                (*OneGates)[Index5D(OneShape, pTime, ba, 0, 0, h+i)]   = (*tempZGates)[Index5D(EachShape, pTime, ba, 0, 0, i)];
            }
        }

        m_aAddgBias->BackPropagate(pTime);
        m_aAddGates->BackPropagate(pTime);
        m_aMatMulH2RZ->BackPropagate(pTime);
        m_aMatMulI2G->BackPropagate(pTime);

        if(pTime == 0 && m_pInitHidden != NULL){

            m_aMatMulH2RZ->BackPropagate(pTime);
            m_aRAndHidden->BackPropagate(pTime);
            m_aBeforeZHidden->BackPropagate(pTime);

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
     /**
     * @brief GPU에서 동작하는 ForwardPropagate 메소드
     * @details cudnn이 제공하는 GRU ForwardPropagate 메소드를 실행한다.
     * @param pTime 연산 할 Tensor가 위치한 Time값.
     * @return int 
     */
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

    /**
     * @brief GPU에서 동작하는 BackwardPropagate 메소드.
     * @details cudnn이 제공하는 GRU BackwardPropagate 메소드를 실행한다.
     * @details 계산한 Gradient는 input_gradient, weight_gradient에 저장한다.
     * @param pTime 연산 할 Tensor가 위치한 Time값.
     * @return int 
     */
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

    /**
     * @brief 맴버변수로 갖고있는 Operator들의 Result값을 Reset시킨다.
     * 
     * @return int 
     */
    int ResetResult() {

        //matmul, bias
        m_aMatMulI2G->ResetResult();
        m_aMatMulH2RZ->ResetResult();
        m_aAddGates->ResetResult();
        m_aAddgBias->ResetResult();

        //R Gate
        m_aRGateInput->ResetResult();
        m_aRGateSigmoid->ResetResult();

        //Z Gate
        m_aZGateInput->ResetResult();
        m_aZGateSigmoid->ResetResult();

        //Candidate Hidden
        m_aMatMulI2CH->ResetResult();
        m_aRAndHidden->ResetResult();
        m_aMatMulH2CH->ResetResult();
        m_aBeforeCandidateHiddenInput->ResetResult();
        m_aCandidateHiddenInput->ResetResult();
        m_aCandidateHiddenTanh->ResetResult();

        //Hidden state
        m_aBeforeZHidden->ResetResult();
        m_aBeforeGHidden1->ResetResult();
        m_aBeforeGHidden2->ResetResult();
        m_aHidden->ResetResult();

        //time
        m_aTempHidden->ResetResult();

        //initHidden
        if(m_pInitHidden != NULL)
           m_pInitHidden->ResetResult();

        Tensor<DTYPE> *result = this->GetResult();
        result->Reset();

        return TRUE;
    }

    /**
     * @brief 맴버변수로 갖고있는 Operator들의 Gradient값을 Reset시킨다.
     * 
     * @return int 
     */
    int ResetGradient() {

        //matmul, bias
        m_aMatMulI2G->ResetGradient();
        m_aMatMulH2RZ->ResetGradient();
        m_aAddGates->ResetGradient();
        m_aAddgBias->ResetGradient();

        //R Gate
        m_aRGateInput->ResetGradient();
        m_aRGateSigmoid->ResetGradient();

        //Z Gate
        m_aZGateInput->ResetGradient();
        m_aZGateSigmoid->ResetGradient();

        //Candidate Hidden
        m_aMatMulI2CH->ResetGradient();
        m_aRAndHidden->ResetGradient();
        m_aMatMulH2CH->ResetGradient();
        m_aBeforeCandidateHiddenInput->ResetGradient();
        m_aCandidateHiddenInput->ResetGradient();
        m_aCandidateHiddenTanh->ResetGradient();

        //Hidden state
        m_aBeforeZHidden->ResetGradient();
        m_aBeforeGHidden1->ResetGradient();
        m_aBeforeGHidden2->ResetGradient();
        m_aHidden->ResetGradient();

        //time
        m_aTempHidden->ResetGradient();

        //initHidden
        if(m_pInitHidden != NULL)
           m_pInitHidden->ResetGradient();

        Tensor<DTYPE> *grad = this->GetGradient();
        grad->Reset();

        return TRUE;
    }

};

#endif  // GRU_H_
