#ifndef RECURRENT_H_
#define RECURRENT_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

#define TIME     0
#define BATCH    1

/*!
@class Recurrent Recurrent class
*/
template<typename DTYPE> class Recurrent : public Operator<DTYPE>{
private:
    Operator<DTYPE> *m_aInput2Hidden;
    Operator<DTYPE> *m_aHidden2Hidden;
    Operator<DTYPE> *m_aTempHidden;
    Operator<DTYPE> *m_aPrevActivate;
    Operator<DTYPE> *m_aApplyActivation;
    Operator<DTYPE> *m_aAddBias;

    //initHidden
    Operator<DTYPE> *m_aInitHidden;


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
    DTYPE *x;      // input
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
    /**
     * @brief Recurrent의 생성자
     * @details 파라미터로 받은 pInput, pWeightIH, pWeightIH, pBias, pInitHidden로 Alloc한다.
     * @param pInput RNN의 input Operator
     * @param pWeightIH Input에서 hidden으로 가는 Matmul의 weight
     * @param pWeightHH hidden에서 hidden으로 가는 matmul의 weight
     * @param pBias     RNN의 bias
     * @param pInitHidden RNN의 init hidden
     */
    Recurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *pBias, Operator<DTYPE>* pInitHidden = NULL) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, pBias) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, pBias, pInitHidden);
    }

    /**
     * @brief Recurrent의 생성자
     * @details 파라미터로 받은 pInput, pWeightIH, pWeightIH, pBias, pInitHidden로 Alloc한다.
     * @param pInput RNN의 input Operator
     * @param pWeightIH Input에서 hidden으로 가는 Matmul의 weight
     * @param pWeightHH hidden에서 hidden으로 가는 matmul의 weight
     * @param pBias  RNN의 bias
     * @param pName 사용자가 부여한 Operator이름.
     * @param pInitHidden RNN의 init hidden
     */
    Recurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *pBias, std::string pName, Operator<DTYPE>* pInitHidden = NULL) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, pBias, pName) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, pBias, pInitHidden);
    }

    /**
     * @brief Destroy the Recurrent object
     * 
     */
    ~Recurrent() {
        #if __DEBUG__
        std::cout << "Recurrent::~Recurrent()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    /**
     * @brief 파라미터로 받은 pInput, pWeightIH, pWeightIH, pBias, pInitHidden으로 맴버 변수들을 초기화 한다.
     * @details RNN연산에 필요한 MatMul, add, tanh operator를 맴버 변수로 생성한다.
     * @param pInput RNN의 input Operator
     * @param pWeightIH Input에서 hidden으로 가는 Matmul의 weight
     * @param pWeightHH hidden에서 hidden으로 가는 matmul의 weight
     * @param pBias  RNN의 bias
     * @param pInitHidden RNN의 init hidden
     * @return int 
     */
    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *pBias, Operator<DTYPE>* pInitHidden) {

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightXHShape = pWeightIH->GetResult()->GetShape();

        int hidTimeSize  = (*InputShape)[TIME];
        int hidBatchSize = (*InputShape)[BATCH];
        int hidColSize   = (*WeightXHShape)[3];

        m_aInput2Hidden  = new MatMul<DTYPE>(pWeightIH, pInput, "rnn_matmul_xh");
        m_aTempHidden    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempHidden");
        m_aHidden2Hidden = new MatMul<DTYPE>(pWeightHH, m_aTempHidden, "rnn_matmul_hh");
        m_aPrevActivate  = new Addall<DTYPE>(m_aInput2Hidden, m_aHidden2Hidden, "rnn_addall");
        m_aAddBias = new AddColWise<DTYPE>(m_aPrevActivate, pBias, "net_with_bias_");
        m_aApplyActivation  = new Tanh<DTYPE>(m_aAddBias, "rnn_tanh");

        //for initHidden
        m_aInitHidden = pInitHidden;

        //For AnalyzeGraph
        pInput->GetOutputContainer()->Pop(m_aInput2Hidden);
        pBias->GetOutputContainer()->Pop(m_aAddBias);
        pWeightIH->GetOutputContainer()->Pop(m_aInput2Hidden);
        pWeightHH->GetOutputContainer()->Pop(m_aHidden2Hidden);

        Shape *ResultShape = m_aApplyActivation->GetResult()->GetShape();

        int timeSize  = (*ResultShape)[TIME];
        int batchSize = (*ResultShape)[BATCH];
        int colSize   = (*ResultShape)[4];

        this->SetResult(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));
        this->SetGradient(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));

        return TRUE;
    }

#if __CUDNN__
      /**
       * @brief cudnn을 사용하기 전 관련 맴버변수들을 초기화 한다.
       * @details TensorDesriptor들을 생성하고, TensorDesriptor들의 데이터가 batch, channel, row, col 순서로 배치되도록 지정한다.
       * @details RNN연산에 필요한 알고리즘을 정의하고, 연산에 필요한 메모리공간을 할당 받는다.
       * @param idOfDevice 사용할 GPU의 id
       */
      void InitializeAttributeForGPU(unsigned int idOfDevice) {

        Operator<DTYPE> *pInput    = this->GetInput()[0];
        Operator<DTYPE> *pWeightIH = this->GetInput()[1];

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightXHShape = pWeightIH->GetResult()->GetShape();

        int colsizeOfInput = (*InputShape)[4];

        int hidTimeSize    = (*InputShape)[TIME];
        int hidBatchSize   = (*InputShape)[BATCH];
        int hidChannelSize = (*InputShape)[2];
        int hidColSize     = (*WeightXHShape)[3];

        // std::cout<<"hidTimeSize : "<<hidTimeSize<<'\n';
        // std::cout<<"hidBatchSize : "<<hidBatchSize<<'\n';
        // std::cout<<"hidColSize : "<<hidColSize<<'\n';
        // std::cout<<"colsizeOfInput : "<<colsizeOfInput<<'\n';

        if(m_aInitHidden != NULL)
         m_aInitHidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);


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
        rnn_mode = CUDNN_RNN_TANH;
        rnn_algo = CUDNN_RNN_ALGO_STANDARD;

        checkCUDNN(cudnnCreateRNNDescriptor(&rnn_desc));
        checkCUDNN(cudnnSetRNNDescriptor_v6(this->GetCudnnHandle(), rnn_desc, hidColSize, 1, dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, rnn_mode, rnn_algo, CUDNN_DATA_FLOAT));

//Bias
        //checkCUDNN(cudnnSetRNNBiasMode(rnn_desc, CUDNN_RNN_NO_BIAS));
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

       if(m_aInitHidden != NULL){
          checkCudaErrors(cudaMalloc((DTYPE**)&hx, hidColSize * hidBatchSize * sizeof(DTYPE)));
          checkCudaErrors(cudaMalloc((DTYPE**)&dhx, hidColSize * hidBatchSize * sizeof(DTYPE)));
        }

       checkCudaErrors(cudaMalloc((DTYPE**)&weights, weight_size));
       checkCudaErrors(cudaMalloc((DTYPE**)&gweights, weight_size));

      }

#endif  // if __CUDNN__

    void Delete() {}

    /**
     * @brief RNN의 ForwardPropagate 메소드
     * @details RNN 연산 중 Input to hidden, hidden to hidden, activation function인 tanh 연산을 수행한다.
     * @details tempHidden이라는 Operator를 사용하여 t-1 에서 t로 가는 hidden 2 hidden 연산을 수행한다.
     * @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
     * @return int 
     */
    int  ForwardPropagate(int pTime = 0) {

        if(pTime==0 && m_aInitHidden != NULL){

            Tensor<DTYPE> *initHidden = m_aInitHidden->GetResult();
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

        m_aInput2Hidden->ForwardPropagate(pTime);

        if (pTime != 0) {
            Tensor<DTYPE> *prevHidden = m_aApplyActivation->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            int batchsize      = prevHidden->GetBatchSize();
            int colSize        = prevHidden->GetColSize();
            Shape *HiddenShape = prevHidden->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int i = 0; i < colSize; i++) {
                    (*tempHidden)[Index5D(HiddenShape, pTime, ba, 0, 0, i)] = (*prevHidden)[Index5D(HiddenShape, pTime - 1, ba, 0, 0, i)];
                }
            }

            m_aHidden2Hidden->ForwardPropagate(pTime);

        }

        m_aPrevActivate->ForwardPropagate(pTime);

        m_aAddBias->ForwardPropagate(pTime);

        m_aApplyActivation->ForwardPropagate(pTime);

        Tensor<DTYPE> *_result = m_aApplyActivation->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        int batchsize      = result->GetBatchSize();
        Shape *ResultShape = result->GetShape();

        for(int ba = 0; ba<batchsize; ba++){
            for (int i = 0; i < colSize; i++) {
                (*result)[Index5D(ResultShape, pTime, ba, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, ba, 0, 0, i)];
            }
        }

        return TRUE;
    }

    /**
     * @brief Recurrent의 BackPropagate 메소드
     * @details RNN 연산의 미분값을 계산하여 input_delta, WeightIH_delta, WeightHH_delta, Bias_delta, InitHidden_delta에 각각 더해 넣는다.
     * @details tempHidden이라는 Operator를 사용하여 t+1 에서 t로 가는 hidden 2 hidden에 해당하는 Gradient 전달하여준다.
     * @param pTime 연산 할 Tensor가 위치한 Time값. default는 0을 사용.
     * @return int 
     */
    int BackPropagate(int pTime = 0) {

        Tensor<DTYPE> *_grad = m_aApplyActivation->GetGradient();
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
            m_aHidden2Hidden->BackPropagate(pTime+1);

            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
            Tensor<DTYPE> *prevHiddenGrad = m_aApplyActivation->GetGradient();

            int colSize        = tempHiddenGrad->GetColSize();
            Shape *HiddenShape = tempHiddenGrad->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int i = 0; i < colSize; i++) {
                    (*prevHiddenGrad)[Index5D(HiddenShape, pTime, ba, 0, 0, i)] += (*tempHiddenGrad)[Index5D(HiddenShape, pTime+1, ba, 0, 0, i)];
                }
            }
        }

        m_aApplyActivation->BackPropagate(pTime);

        m_aAddBias->BackPropagate(pTime);

        m_aPrevActivate->BackPropagate(pTime);

        m_aInput2Hidden->BackPropagate(pTime);

        if(pTime == 0 && m_aInitHidden != NULL){

            m_aHidden2Hidden->BackPropagate(pTime);

            Tensor<DTYPE> *initHiddenGrad = m_aInitHidden->GetGradient();
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
     * @details cudnn이 제공하는 RNN ForwardPropagate 메소드를 실행한다.
     * @param pTime 연산 할 Tensor가 위치한 Time값.
     * @return int 
     */
    int ForwardPropagateOnGPU(int pTime = 0) {

        Tensor<DTYPE> *input    = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *weightHH = this->GetInput()[2]->GetResult();
        Tensor<DTYPE> *result   = this->GetResult();

        //
      //  DTYPE *wicwiuX       = input->GetGPUData(pTime);
        weights = weightHH->GetGPUData(0);
      //  DTYPE *wicwiuY       = result->GetGPUData(pTime);

        int timeSize        = input->GetTimeSize();

        //mode = 0 : train
        // if(this->GetMode() == 0 && pTime != timeSize-1)                   //여기 주석처리해도 문제는 안생김.... 다만 느려질 뿐...!
        //   return TRUE;

        cx = NULL;
        cy = NULL;
        hy = NULL;

        if(m_aInitHidden != NULL){

          Tensor<DTYPE> *initHidden = m_aInitHidden->GetResult();
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
     * @details cudnn이 제공하는 RNN BackwardPropagate 메소드를 실행한다.
     * @details 계산한 Gradient는 input_gradient, weightHH_gradient에 저장한다.
     * @param pTime 연산 할 Tensor가 위치한 Time값.
     * @return int 
     */
    int BackPropagateOnGPU(int pTime = 0) {

        Tensor<DTYPE> *input             = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta       = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *weightHH          = this->GetInput()[2]->GetResult();
        Tensor<DTYPE> *weightHH_gradient = this->GetInput()[2]->GetGradient();
        Tensor<DTYPE> *result            = this->GetResult();
        Tensor<DTYPE> *this_delta        = this->GetDelta();


        weights     = weightHH->GetGPUData(0);
        gweights    = weightHH_gradient->GetGPUData(0);

        int timeSize  = input->GetTimeSize();


        if(pTime != 0)
          return TRUE;

        dhy = NULL;
        dcy = NULL;
        cx = NULL;
        dcx = NULL;

        if(m_aInitHidden != NULL){

          Tensor<DTYPE> *initHidden = m_aInitHidden->GetResult();
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

        if(m_aInitHidden != NULL){

          Tensor<DTYPE> *initHidden = m_aInitHidden->GetGradient();
          int m_Capacity = initHidden->GetCapacity();               //time = 0이니깐
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
        m_aInput2Hidden->ResetResult();
        m_aHidden2Hidden->ResetResult();
        m_aTempHidden->ResetResult();
        m_aPrevActivate->ResetResult();
        m_aApplyActivation->ResetResult();
        m_aAddBias->ResetResult();

        Tensor<DTYPE> *result = this->GetResult();
        result->Reset();

        //initHidden
        if(m_aInitHidden != NULL)
           m_aInitHidden->ResetResult();

         return TRUE;

    }

    /**
     * @brief 맴버변수로 갖고있는 Operator들의 Gradient값을 Reset시킨다.
     * 
     * @return int 
     */
    int ResetGradient() {
        m_aInput2Hidden->ResetGradient();
        m_aHidden2Hidden->ResetGradient();
        m_aTempHidden->ResetGradient();
        m_aPrevActivate->ResetGradient();
        m_aApplyActivation->ResetGradient();
        m_aAddBias->ResetGradient();

        Tensor<DTYPE> *grad = this->GetGradient();
        grad->Reset();

        //initHidden
        if(m_aInitHidden != NULL)
           m_aInitHidden->ResetGradient();

         return TRUE;

    }


};


#endif  // RECURRENTCUDNN_H_
