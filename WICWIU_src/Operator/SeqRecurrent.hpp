#ifndef SEQRECURRENT_H_
#define SEQRECURRENT_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

#define TIME     0
#define BATCH    1

//인터넷 방식 + 전체 time 처리

template<typename DTYPE> class SeqRecurrent : public Operator<DTYPE>{
private:
    Operator<DTYPE> *m_aInput2Hidden;
    Operator<DTYPE> *m_aHidden2Hidden;
    Operator<DTYPE> *m_aTempHidden;
    Operator<DTYPE> *m_aPrevActivate;
    Operator<DTYPE> *ApplyActivation;
    Operator<DTYPE> *AddBias;

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
    SeqRecurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias, Operator<DTYPE>* pInitHidden = NULL) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, rBias) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, rBias, pInitHidden);
    }

    //pName때문에 Operator 생성자 호출이 안되는듯!!!!   숫자 4로해도 되는건가?
    SeqRecurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias, std::string pName, Operator<DTYPE>* pInitHidden = NULL) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, rBias, pName) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, rBias, pInitHidden);
    }

    ~SeqRecurrent() {
        #if __DEBUG__
        std::cout << "Recurrent::~Recurrent()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias, Operator<DTYPE>* pInitHidden) {

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightXHShape = pWeightIH->GetResult()->GetShape();

        int hidTimeSize  = (*InputShape)[TIME];
        int hidBatchSize = (*InputShape)[BATCH];
        int hidColSize   = (*WeightXHShape)[3];

        m_aInput2Hidden  = new MatMul<DTYPE>(pWeightIH, pInput, "rnn_matmul_xh");
        m_aTempHidden    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempHidden");
        m_aHidden2Hidden = new MatMul<DTYPE>(pWeightHH, m_aTempHidden, "rnn_matmul_hh");
        m_aPrevActivate  = new Addall<DTYPE>(m_aInput2Hidden, m_aHidden2Hidden, "rnn_addall");
        AddBias = new AddColWise<DTYPE>(m_aPrevActivate, rBias, "net_with_bias_");
        ApplyActivation  = new Tanh<DTYPE>(AddBias, "rnn_tanh");

        //for initHidden
        m_aInitHidden = pInitHidden;

        //ApplyActivation  = new Relu<DTYPE>(AddBias, "rnn_tanh");

        //For AnalyzeGraph
        pInput->GetOutputContainer()->Pop(m_aInput2Hidden);   // 21년 3월 4일 추가!
        rBias->GetOutputContainer()->Pop(AddBias);
        pWeightIH->GetOutputContainer()->Pop(m_aInput2Hidden);
        pWeightHH->GetOutputContainer()->Pop(m_aHidden2Hidden);

        Shape *ResultShape = ApplyActivation->GetResult()->GetShape();

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

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightXHShape = pWeightIH->GetResult()->GetShape();

        int colsizeOfInput = (*InputShape)[4];

        int hidTimeSize    = (*InputShape)[TIME];
        int hidBatchSize   = (*InputShape)[BATCH];
        //int hidBatchSize   = 1;
        int hidChannelSize = (*InputShape)[2];
        int hidColSize     = (*WeightXHShape)[3];


        std::cout<<"hidTimeSize : "<<hidTimeSize<<'\n';
        std::cout<<"hidBatchSize : "<<hidBatchSize<<'\n';
        std::cout<<"hidColSize : "<<hidColSize<<'\n';
        std::cout<<"colsizeOfInput : "<<colsizeOfInput<<'\n';

        //inithidden 값을 저장하고 있는 operator도  GPU로 올려야 될거 같음!
        if(m_aInitHidden != NULL)
         m_aInitHidden->SetDeviceGPU(this->GetCudnnHandle(), idOfDevice);          // --> 문제있으면 여기 하기!!!    ??? 왜 이걸 추가하면 segfault가 뜨는걸까...

        std::cout<<"호출 완료"<<'\n';

        x_desc = new cudnnTensorDescriptor_t[hidTimeSize];
        dx_desc = new cudnnTensorDescriptor_t[hidTimeSize];
        y_desc = new cudnnTensorDescriptor_t[hidTimeSize];
        dy_desc = new cudnnTensorDescriptor_t[hidTimeSize];


//tensor descriptor
        //배열형태인 tensor descriptor


        for(int i=0; i<hidTimeSize; i++){
            checkCUDNN(cudnnCreateTensorDescriptor(&x_desc[i]));
            checkCUDNN(cudnnCreateTensorDescriptor(&dx_desc[i]));

            checkCUDNN(cudnnCreateTensorDescriptor(&y_desc[i]));
            checkCUDNN(cudnnCreateTensorDescriptor(&dy_desc[i]));

            //설정
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



        //배열 형태가 아닌 tensor descriptor 생성
        checkCUDNN(cudnnCreateTensorDescriptor(&hx_desc));      //initial hidden state
        checkCUDNN(cudnnCreateTensorDescriptor(&dhx_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&cx_desc));      //initial cell state
        checkCUDNN(cudnnCreateTensorDescriptor(&dcx_desc));

        checkCUDNN(cudnnCreateTensorDescriptor(&hy_desc));      //fianl hidden state
        checkCUDNN(cudnnCreateTensorDescriptor(&dhy_desc));
        checkCUDNN(cudnnCreateTensorDescriptor(&cy_desc));      //final cell state
        checkCUDNN(cudnnCreateTensorDescriptor(&dcy_desc));


        //배열 형태가 아닌 tensor descriptor 설정
        //일단은 4D함수를 사용해서 하고, batch, channel, row, col로 함!!!
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
        checkCUDNN(cudnnDropoutGetStatesSize(this->GetCudnnHandle(), &state_size));    //state size구하기
        checkCUDNN(cudnnSetDropoutDescriptor(dropout_desc, this->GetCudnnHandle(), m_droprate, state, state_size, time(NULL))); //일단은 NULL로 줌!


//rnn descriptor
        rnn_mode = CUDNN_RNN_TANH;
        rnn_algo = CUDNN_RNN_ALGO_STANDARD;

        checkCUDNN(cudnnCreateRNNDescriptor(&rnn_desc));
        //version6로 안해도될듯? 음... 잘 모르겠다
        //numLayers : number of stacked layers = 1 로 설정해줌
        checkCUDNN(cudnnSetRNNDescriptor_v6(this->GetCudnnHandle(), rnn_desc, hidColSize, 1, dropout_desc, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, rnn_mode, rnn_algo, CUDNN_DATA_FLOAT));

//Bias
        //checkCUDNN(cudnnSetRNNBiasMode(rnn_desc, CUDNN_RNN_NO_BIAS));
        checkCUDNN(cudnnSetRNNBiasMode(rnn_desc, CUDNN_RNN_SINGLE_INP_BIAS));

//filter descriptor

        //생성
        checkCUDNN(cudnnCreateFilterDescriptor(&w_desc));
        checkCUDNN(cudnnCreateFilterDescriptor(&dw_desc));

        //설정

            //weight size 받아오기
        //237pg
        checkCUDNN(cudnnGetRNNParamsSize(this->GetCudnnHandle(), rnn_desc, x_desc[0], &weight_size, CUDNN_DATA_FLOAT));

        dimW[0] = weight_size / sizeof(float);
        dimW[1] = 1;
        dimW[2] = 1;

        std::cout<<"-------------------------------------------------------------------------"<<'\n';
        std::cout<<"weight_size : "<<weight_size<<'\n';
        std::cout<<"dimW[0] : "<<weight_size / sizeof(float)<<'\n';

        checkCUDNN(cudnnSetFilterNdDescriptor(w_desc,  CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));
        checkCUDNN(cudnnSetFilterNdDescriptor(dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));







//workspace    hidTimeSize
        //240pg
        checkCUDNN(cudnnGetRNNWorkspaceSize(this->GetCudnnHandle(), rnn_desc,
                                            hidTimeSize,
                                            x_desc,
                                            &workspace_size));   //seqLength 정확하지 않음, 240pg

        checkCUDNN(cudnnGetRNNTrainingReserveSize(this->GetCudnnHandle(), rnn_desc, hidTimeSize, x_desc, &reserved_size));       //seqLength 정확하지 않음, 239pg

  //      checkCudaErrors(cudaMalloc(&workspace, workspace_size));
  //      checkCudaErrors(cudaMalloc(&reserved_space, reserved_size));


        //std::cout<<"reserved_size : "<<reserved_size<<'\n';

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

        //sequence 전체를 한번에 처리하기 위해!
        int input_length = hidTimeSize * colsizeOfInput * hidBatchSize;
        int output_length = hidTimeSize * hidColSize * hidBatchSize;

       checkCudaErrors(cudaMalloc((DTYPE**)&x, input_length * sizeof(DTYPE)));
       checkCudaErrors(cudaMalloc((DTYPE**)&dx, input_length * sizeof(DTYPE)));
       checkCudaErrors(cudaMalloc((DTYPE**)&y, output_length * sizeof(DTYPE)));
       checkCudaErrors(cudaMalloc((DTYPE**)&dy, output_length * sizeof(DTYPE)));

       //hx, cx와 같은 부분은 필요 없는 듯
       //hx즉 inithidden값이 있을 때 설정해주기!!!
       if(m_aInitHidden != NULL){
          checkCudaErrors(cudaMalloc((DTYPE**)&hx, hidColSize * hidBatchSize * sizeof(DTYPE)));
          checkCudaErrors(cudaMalloc((DTYPE**)&dhx, hidColSize * hidBatchSize * sizeof(DTYPE)));
        }


       //dw는 어떻게 하지?... -> 이거는 time마다 있는게 아니니깐 그냥 wicwiu방식으로 하면 될 듯?
       //weight도 해보자!
       checkCudaErrors(cudaMalloc((DTYPE**)&weights, weight_size));
       checkCudaErrors(cudaMalloc((DTYPE**)&gweights, weight_size));



      }

#endif  // if __CUDNN__

    //이거 해줘야되나?
    void Delete() {}


    int  ForwardPropagate(int pTime = 0) {


        //*********************************************inithidden 처리하기!!!*****************************************
        if(pTime==0 && m_aInitHidden != NULL){

            // std::cout<<"SeqRecurrent init hidden 처리"<<'\n';

            Tensor<DTYPE> *initHidden = m_aInitHidden->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            int colsize      = initHidden->GetColSize();
            int batchsize    = initHidden->GetBatchSize();
            //2개의 shape은 다르지!!!
            Shape *initShape = initHidden->GetShape();
            Shape *tempShape = tempHidden->GetShape();

            // std::cout<<initShape<<'\n';
            // std::cout<<tempShape<<'\n';

            for(int ba=0; ba<batchsize; ba++){
                for (int co = 0; co < colsize; co++){
                    (*tempHidden)[Index5D(tempShape, pTime, ba, 0, 0, co)] = (*initHidden)[Index5D(initShape, pTime, ba, 0, 0, co)];
                }
            }

            //std::cout<<"inithidden값!"<<'\n';
            //std::cout<<initHidden<<'\n';
        }
        //*********************************************inithidden 처리하기!!!*****************************************


        m_aInput2Hidden->ForwardPropagate(pTime);

        if (pTime != 0) {
            Tensor<DTYPE> *prevHidden = ApplyActivation->GetResult();
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

        AddBias->ForwardPropagate(pTime);

        ApplyActivation->ForwardPropagate(pTime);

        Tensor<DTYPE> *_result = ApplyActivation->GetResult();
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

    int BackPropagate(int pTime = 0) {


        Tensor<DTYPE> *_grad = ApplyActivation->GetGradient();
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
            Tensor<DTYPE> *prevHiddenGrad = ApplyActivation->GetGradient();

            int colSize        = tempHiddenGrad->GetColSize();
            Shape *HiddenShape = tempHiddenGrad->GetShape();

            for(int ba=0; ba<batchsize; ba++){
                for (int i = 0; i < colSize; i++) {
                    (*prevHiddenGrad)[Index5D(HiddenShape, pTime, ba, 0, 0, i)] += (*tempHiddenGrad)[Index5D(HiddenShape, pTime+1, ba, 0, 0, i)];
                }
            }
        }

        //std::cout<<"Gradient 앞에 time꺼 잘 갖고왔는가"<<'\n';
        //std::cout<<ApplyActivation->GetGradient()<<'\n';
        ApplyActivation->BackPropagate(pTime);

        AddBias->BackPropagate(pTime);

        m_aPrevActivate->BackPropagate(pTime);

        m_aInput2Hidden->BackPropagate(pTime);

        //*********************************************inithidden 처리하기!!!*****************************************
        if(pTime == 0 && m_aInitHidden != NULL){

            //std::cout<<"seqRecurrent Backward init hidden 처리"<<'\n';

            m_aHidden2Hidden->BackPropagate(pTime);

            Tensor<DTYPE> *initHiddenGrad = m_aInitHidden->GetGradient();
            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();

            // std::cout<<"넘겨주려는 gradient값"<<'\n';
            // std::cout<<tempHiddenGrad->GetShape()<<'\n';
            // std::cout<<tempHiddenGrad<<'\n';

            //2개 shape이 다름 !!! time이 다르게 되어있음!!!
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
        //*********************************************inithidden 처리하기!!!*****************************************


        return TRUE;
    }


#if __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {

         // std::cout<<"RNN Forward GPU"<<pTime<<'\n';

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

        //입력 잘 들어감!
        // std::cout<<"Forward"<<'\n';
        // std::cout<<"Recurrent input"<<'\n';
        // std::cout<<input->GetShape()<<'\n';
        // std::cout<<input<<'\n';

        //NULL해도 되는거 처리하기
        cx = NULL;
        cy = NULL;
        // hx = NULL;
        hy = NULL;

        //inithidden값이 있는 경우!!!
        if(m_aInitHidden != NULL){

          Tensor<DTYPE> *initHidden = m_aInitHidden->GetResult();
          int m_Capacity = initHidden->GetCapacity();               //time = 0이니깐
          DTYPE * wicwiuInitHidden = initHidden->GetGPUData(0);

          checkCudaErrors(cudaMemcpy(&hx[0], wicwiuInitHidden, (m_Capacity * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));         //여기서 error...
        }
        else{
          hx = NULL;
        }

        //입력은 복사
        int m_CapacityPerTime = input->GetCapacity() / timeSize;

        for(int i=0; i<timeSize; i++){

            DTYPE *wicwiuX       = input->GetGPUData(i);
            checkCudaErrors(cudaMemcpy(&x[m_CapacityPerTime*i], wicwiuX, (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
        }

        //api 300pg
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

        //출력 : y   //출력값을 받은 후 다시 WICWIU에 값 복사해주고!
        m_CapacityPerTime = result->GetCapacity() / timeSize;

        for(int i=0; i<timeSize; i++){
            DTYPE *wicwiuY       = result->GetGPUData(i);
            checkCudaErrors(cudaMemcpy(wicwiuY, &y[m_CapacityPerTime*i], (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));

        }

        // std::cout<<"RNN 결과값 출력 : "<<pTime<<'\n';
        // std::cout<<result<<'\n';

        return TRUE;
    }


    int BackPropagateOnGPU(int pTime = 0) {

        Tensor<DTYPE> *input             = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta       = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *weightHH          = this->GetInput()[2]->GetResult();
        Tensor<DTYPE> *weightHH_gradient = this->GetInput()[2]->GetGradient();
        Tensor<DTYPE> *result            = this->GetResult();
        Tensor<DTYPE> *this_delta        = this->GetDelta();


        weights     = weightHH->GetGPUData(0);
        gweights = weightHH_gradient->GetGPUData(0);

        int timeSize        = input->GetTimeSize();


        //backward에서는... 음... gradient가 +=으로 저장되기 때문에 이걸 없애면 NAN값이 발생하는거 같음!

        if(pTime != 0)                                                           //... 여기 embedding이랑 관련해서 문제 있을거 같은데....
          return TRUE;

      // std::cout<<"Backward"<<'\n';
      // std::cout<<"Recurrent input"<<'\n';
      // std::cout<<input->GetShape()<<'\n';
      // std::cout<<input<<'\n';

      // std::cout<<"Recurrent gradient"<<'\n';
      // std::cout<<this_delta->GetShape()<<'\n';
      // std::cout<<this_delta<<'\n';


        dhy = NULL;
        dcy = NULL;
        cx = NULL;
        dcx = NULL;
        // hx = NULL;
        // dhx = NULL;

        //inithidden값이 있는 경우!!!
        if(m_aInitHidden != NULL){

          // std::cout<<"GPU Back initHidden 처리"<<'\n';

          Tensor<DTYPE> *initHidden = m_aInitHidden->GetResult();
          int m_Capacity = initHidden->GetCapacity();               //time = 0이니깐
          DTYPE * wicwiuInitHidden = initHidden->GetGPUData(0);

          checkCudaErrors(cudaMemcpy(&hx[0], wicwiuInitHidden, (m_Capacity * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
        }
        else
        {
            hx = NULL;
            dhx = NULL;
        }

        //y, dy 하나의 sequence로 만들어서 넣어주기!

        int m_CapacityPerTime = result->GetCapacity() / timeSize;

        for(int i=0; i<timeSize; i++){
            DTYPE *wicwiuY             = result->GetGPUData(i);
            DTYPE *wicwiuY_delta       = this_delta->GetGPUData(i);
            checkCudaErrors(cudaMemcpy(&y[m_CapacityPerTime*i], wicwiuY, (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(&dy[m_CapacityPerTime*i], wicwiuY_delta, (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));

        }

        //weight 하나로 펴서 넣어주기!

        // std::cout<<"weight capacity : "<<weightHH->GetCapacity()<<'\n';

        // int weight_row = weightHH->GetRowSize();
        // int weight_col = weightHH->GetColSize();
        //
        // for(int i=0; i<weight_row; i++){
        //
        //     checkCudaErrors(cudaMemcpy(&weights[weight_col*i], wicwiuY, (weight_col * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
        //
        // }


        //276pg
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

        //dx값 원래 WICWIU로 복사해주기!
        m_CapacityPerTime = input->GetCapacity() / timeSize;

        for(int i=0; i<timeSize; i++){
            DTYPE *wicwiuX_delta       = input_delta->GetGPUData(i);
            checkCudaErrors(cudaMemcpy(wicwiuX_delta, &dx[m_CapacityPerTime*i], (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));

        }

        // //이걸 출력해본 결과....! 값이 변경되면 안되는데.... 값이 변경되고 있음!!!... 아주아주 조금씩 축적되는게 맞는거 같음!
        // std::cout<<"값 축적인지 확인하기"<<'\n';
        // std::cout<<input_delta->GetShape()<<'\n';
        // std::cout<<input_delta<<'\n';

        //inithidden값이 있는 경우 --> dhx값을 inithidden gradient에 넣어주기!
        if(m_aInitHidden != NULL){

          Tensor<DTYPE> *initHidden = m_aInitHidden->GetGradient();
          int m_Capacity = initHidden->GetCapacity();               //time = 0이니깐
          DTYPE * wicwiuInitHidden = initHidden->GetGPUData(0);

          checkCudaErrors(cudaMemcpy(wicwiuInitHidden, &dhx[0], (m_Capacity * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
        }


        /////////////////////////////////////////////////////////////////////////////

        //x값 복사해주기!
        m_CapacityPerTime = input->GetCapacity() / timeSize;

        for(int i=0; i<timeSize; i++){

            DTYPE *wicwiuX       = input->GetGPUData(i);
            checkCudaErrors(cudaMemcpy(&x[m_CapacityPerTime*i], wicwiuX, (m_CapacityPerTime * sizeof(DTYPE)), cudaMemcpyDeviceToDevice));
        }

        //286pg
        checkCUDNN(cudnnRNNBackwardWeights(this->GetCudnnHandle(), rnn_desc, timeSize,
                                           x_desc, x,
                                           hx_desc, hx,
                                           y_desc, y,
                                           workspace, workspace_size,
                                           dw_desc, gweights,
                                           reserved_space, reserved_size));

        //dw값 이거는 time별로 있는게 아니라 모든 time에 대해서 동일하니깐 상관없지 않을까???

         // std::cout<<"weight_HH gradient"<<'\n';
         // std::cout<<weightHH_gradient->GetShape()<<'\n';
         // std::cout<<weightHH_gradient<<'\n';

        return TRUE;
    }
#endif  // if __CUDNN__



    // GPU에 대한 Reset 처리는 operator.hpp에 되어있음
    int ResetResult() {
        m_aInput2Hidden->ResetResult();
        m_aHidden2Hidden->ResetResult();
        m_aTempHidden->ResetResult();
        m_aPrevActivate->ResetResult();
        ApplyActivation->ResetResult();
        AddBias->ResetResult();

        Tensor<DTYPE> *result = this->GetResult();
        result->Reset();

        //initHidden
        if(m_aInitHidden != NULL)
           m_aInitHidden->ResetResult();

         return TRUE;

    }

    int ResetGradient() {
        m_aInput2Hidden->ResetGradient();
        m_aHidden2Hidden->ResetGradient();
        m_aTempHidden->ResetGradient();
        m_aPrevActivate->ResetGradient();
        ApplyActivation->ResetGradient();
        AddBias->ResetGradient();

        Tensor<DTYPE> *grad = this->GetGradient();
        grad->Reset();

        //initHidden
        if(m_aInitHidden != NULL)
           m_aInitHidden->ResetGradient();

         return TRUE;

    }


};


#endif  // RECURRENTCUDNN_H_
