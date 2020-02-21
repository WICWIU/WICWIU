#ifndef RECURRENT_H_
#define RECURRENT_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

template<typename DTYPE> class Recurrent : public Operator<DTYPE>{
private:
    Operator<DTYPE> *m_aInput2Hidden;
    Operator<DTYPE> *m_aHidden2Hidden;
    Operator<DTYPE> *m_aTempHidden;
    Operator<DTYPE> *m_aPrevActivate;
    Operator<DTYPE> *m_aPostActivate;
    Operator<DTYPE> *m_aHidden2Output;

public:
  //Recurrent는 지금 생성자에 인자가 너무 많아서 Operator의 생성자를 호출할 때 숫자를 넣어줘야됨 그래서 4를 넣어준거
    Recurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightXH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *pWeightHY) : Operator<DTYPE>(4, pInput, pWeightXH, pWeightHH, pWeightHY) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightXH, pWeightHH, pWeightHY);
    }

    Recurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightXH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *pWeightHY, std::string pName) : Operator<DTYPE>(pInput, pWeightXH, pWeightHH, pWeightHY, pName) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightXH, pWeightHH, pWeightHY, pName);
    }

    ~Recurrent() {
        #if __DEBUG__
        std::cout << "Recurrent::~Recurrent()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightXH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *pWeightHY, std::string pName) {
        #if __DEBUG__
        std::cout << "Recurrent::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        Shape *InputShape    = pInput->GetResult()->GetShape();
        Shape *WeightXHShape = pWeightXH->GetResult()->GetShape();

        // Operator<DTYPE> *ptempWeight_x2h = new Tensorholder<float>(1, 1, 1, 10, 4, "ptempWeight_x2h");
        // Operator<DTYPE> *ptempWeight_h2h = new Tensorholder<float>(1, 1, 1, 10, 10, "ptempWeight_h2h");
        // Operator<DTYPE> *ptempWeight_h2o = new Tensorholder<float>(1, 1, 1, 4, 10, "ptempWeight_h2o");
        //
        // *ptempWeight_x2h = *pWeightXH;
        // *ptempWeight_h2h = *pWeightHH;
        // *ptempWeight_h2o = *pWeightHY;

        int hidTimeSize  = (*InputShape)[0];
        int hidBatchSize = (*InputShape)[1];
        int hidColSize   = (*WeightXHShape)[3];

        m_aInput2Hidden  = new MatMul<DTYPE>(pWeightXH, pInput, "rnn_matmul_xh");
        m_aTempHidden    = new Tensorholder<DTYPE>(Tensor<DTYPE>::Zeros(hidTimeSize, hidBatchSize, 1, 1, hidColSize), "tempHidden");
        m_aHidden2Hidden = new MatMul<DTYPE>(pWeightHH, m_aTempHidden, "rnn_matmul_hh");
        m_aPrevActivate  = new Addall<DTYPE>(m_aInput2Hidden, m_aHidden2Hidden, "rnn_addall");
        m_aPostActivate  = new Tanh<DTYPE>(m_aPrevActivate, "rnn_tanh");
        m_aHidden2Output= new MatMul<DTYPE>(pWeightHY, m_aPostActivate, "rnn_matmul_hh");

        std::cout << "m_aInput2Hidden : " << m_aInput2Hidden->GetResult()->GetShape() << '\n';
        std::cout << "m_aTempHidden : " << m_aTempHidden->GetResult()->GetShape() << '\n';
        std::cout << "m_aHidden2Hidden : " << m_aHidden2Hidden->GetResult()->GetShape() << '\n';
        std::cout << "m_aPrevActivate : " << m_aPrevActivate->GetResult()->GetShape() << '\n';
        std::cout << "m_aPostActivate : " << m_aPostActivate->GetResult()->GetShape() << '\n';
        std::cout << "m_aHidden2Output : " << m_aHidden2Output->GetResult()->GetShape() << '\n';

        pWeightXH->GetOutputContainer()->Pop(m_aInput2Hidden);
        pWeightHH->GetOutputContainer()->Pop(m_aHidden2Hidden);
        pWeightHY->GetOutputContainer()->Pop(m_aHidden2Output);

        Shape *ResultShape = m_aHidden2Output->GetResult()->GetShape();

        int timeSize  = (*ResultShape)[0];
        int batchSize = (*ResultShape)[1];
        int colSize   = (*ResultShape)[4];

        this->SetResult(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));
        this->SetGradient(Tensor<DTYPE>::Zeros(timeSize, batchSize, 1, 1, colSize));

        return TRUE;
    }

#if __CUDNN__
    void InitializeAttributeForGPU() {
        m_aInput2Hidden->SetDeviceGPU();
        m_aInput2Hidden->SetCudnnHandle(this->GetCudnnHandle());

        m_aHidden2Hidden->SetDeviceGPU();
        m_aHidden2Hidden->SetCudnnHandle(this->GetCudnnHandle());

        m_aTempHidden->SetDeviceGPU();
        m_aTempHidden->SetCudnnHandle(this->GetCudnnHandle());

        m_aPrevActivate->SetDeviceGPU();
        m_aPrevActivate->SetCudnnHandle(this->GetCudnnHandle());

        m_aPostActivate->SetDeviceGPU();
        m_aPostActivate->SetCudnnHandle(this->GetCudnnHandle());

        m_aHidden2Output->SetDeviceGPU();
        m_aHidden2Output->SetCudnnHandle(this->GetCudnnHandle());
    }

#endif  // if __CUDNN__

    void Delete() {}

    //int  ForwardPropagate(int pTime = 0, int pThreadNum = 0)
    int  ForwardPropagate(int pTime = 0) {

        m_aInput2Hidden->ForwardPropagate(pTime);
        #if __RNN_DEBUG__
        std::cout << "\n============= m_aInput2Hidden : MatMul<DTYPE>(pWeightXH, pInput) =================\n" << m_aInput2Hidden->GetResult() << "\n";
        #endif

        if (pTime != 0) {
            Tensor<DTYPE> *prevHidden = m_aPostActivate->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            int colSize        = prevHidden->GetColSize();
            Shape *HiddenShape = prevHidden->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*tempHidden)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] = (*prevHidden)[Index5D(HiddenShape, pTime - 1, 0, 0, 0, i)];
            }

            m_aHidden2Hidden->ForwardPropagate(pTime);
            #if __RNN_DEBUG__
            std::cout << "\n============= m_aHidden2Hidden : MatMul<DTYPE>(pWeightHH, m_aTempHidden) =================\n" << m_aHidden2Hidden->GetResult( )<< "\n";
            #endif
        }
        m_aPrevActivate->ForwardPropagate(pTime);
        #if __RNN_DEBUG__
        std::cout << "\n============= m_aPrevActivate : Addall<DTYPE>(m_aInput2Hidden, m_aHidden2Hidden) =================\n" << m_aPrevActivate->GetResult() << "\n";
        #endif

        m_aPostActivate->ForwardPropagate(pTime);
        #if __RNN_DEBUG__
        std::cout << "\n============= m_aPostActivate : Tanh<DTYPE>(m_aPrevActivate) =================\n" << m_aPostActivate->GetResult() << "\n";
        #endif

        m_aHidden2Output->ForwardPropagate(pTime);
        #if __RNN_DEBUG__
        std::cout << "\n============= m_aHidden2Output : MatMul<DTYPE>(pWeightHY, m_aPostActivate) =================\n" << m_aHidden2Output->GetResult() << "\n";
        #endif

        Tensor<DTYPE> *_result = m_aHidden2Output->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        Shape *ResultShape = result->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*result)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
            #if __RNN_DEBUG__
            std::cout << "\n ============= Forward result ==============" << result << '\n';
            #endif
        }

        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *_grad = m_aHidden2Output->GetGradient();
        Tensor<DTYPE> *grad  = this->GetGradient();

        int colSize        = grad->GetColSize();
        int timeSize       = grad->GetTimeSize();
        Shape *ResultShape = grad->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*_grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
            #if __RNN_DEBUG__
            std::cout << "\n ============= Backward result ==============" << _grad << '\n';
            #endif
        }

        if (pTime != timeSize-1) {
            m_aHidden2Hidden->BackPropagate(pTime+1);
            #if __RNN_DEBUG__
            std::cout << "\n============= B m_aHidden2Hidden =================\n" << m_aHidden2Hidden->GetResult() << "\n";
            #endif

            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
            Tensor<DTYPE> *prevHiddenGrad = m_aPostActivate->GetGradient();

            int colSize        = tempHiddenGrad->GetColSize();
            Shape *HiddenShape = tempHiddenGrad->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*prevHiddenGrad)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] = (*tempHiddenGrad)[Index5D(HiddenShape, pTime+1, 0, 0, 0, i)];
            }
        }

        m_aHidden2Output->BackPropagate(pTime);
        #if __RNN_DEBUG__
        std::cout << "\n============= B m_aHidden2Output =================\n" << m_aHidden2Output->GetResult() << "\n";
        #endif

        m_aPostActivate->BackPropagate(pTime);
        #if __RNN_DEBUG__
        std::cout << "\n=============  B m_aPostActivate =================\n" << m_aPostActivate->GetResult() << "\n";
        #endif

        m_aPrevActivate->BackPropagate(pTime);
        #if __RNN_DEBUG__
        std::cout << "\n============= B m_aPrevActivate =================\n" << m_aPrevActivate->GetResult() << "\n";
        #endif

        m_aInput2Hidden->BackPropagate(pTime);
        #if __RNN_DEBUG__
        std::cout << "\n============= B m_aInput2Hidden =================\n" << m_aInput2Hidden->GetResult() << "\n";
        #endif
        
        return TRUE;
    }

#if __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {
        int alpha = 1.f;
        int beta  = 0.f;

        cudnnTensorDescriptor_t desc = NULL;

        m_aInput2Hidden->ForwardPropagateOnGPU(pTime);

        if (pTime != 0) {
            Tensor<DTYPE> *prevHidden = m_aHidden2Hidden->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            DTYPE *pDevPrevHidden = prevHidden->GetDeviceData(pTime - 1);
            DTYPE *pDevTempHidden = tempHidden->GetDeviceData(pTime);

            desc = prevHidden->GetDescriptor();

            checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                      &alpha, desc, pDevPrevHidden,
                                      &alpha, desc, pDevTempHidden));

            m_aHidden2Hidden->ForwardPropagateOnGPU(pTime);
        }

        m_aPrevActivate->ForwardPropagateOnGPU(pTime);

        m_aPostActivate->ForwardPropagateOnGPU(pTime);

        m_aHidden2Output->ForwardPropagateOnGPU(pTime);

        Tensor<DTYPE> *_result = m_aHidden2Output->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        Shape *ResultShape = result->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*result)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }

        // DTYPE *_pDevResult = _result->GetDeviceData(pTime);
        // DTYPE *pDevResult  = result->GetDeviceData(pTime);

        // desc = _result->GetDescriptor();
        //
        // checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
        // &alpha, desc, _pDevResult,
        // &alpha, desc, pDevResult));


        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0) {
        int alpha = 1.f;
        int beta  = 0.f;

        cudnnTensorDescriptor_t desc = NULL;

        Tensor<DTYPE> *_grad = m_aHidden2Output->GetGradient();
        Tensor<DTYPE> *grad  = this->GetGradient();

        DTYPE *_pDevGrad = _grad->GetDeviceData(pTime);
        DTYPE *pDevGrad  = grad->GetDeviceData(pTime);

        int colSize        = grad->GetColSize();
        Shape *ResultShape = grad->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*_grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }

        // std::cout << _grad << '\n';

        // desc = _grad->GetDescriptor();
        //
        // checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
        //                           &alpha, desc, pDevGrad,
        //                           &beta, desc, _pDevGrad));

        m_aHidden2Output->BackPropagateOnGPU(pTime);

        m_aPostActivate->BackPropagateOnGPU(pTime);

        m_aPrevActivate->BackPropagateOnGPU(pTime);

        if (pTime != 0) {
            m_aHidden2Hidden->BackPropagateOnGPU(pTime);

            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
            Tensor<DTYPE> *prevHiddenGrad = m_aHidden2Hidden->GetGradient();

            DTYPE *pDevTempHiddenGrad = tempHiddenGrad->GetDeviceData(pTime);
            DTYPE *pDevPrevHiddenGrad = prevHiddenGrad->GetDeviceData(pTime - 1);

            desc = tempHiddenGrad->GetDescriptor();

            checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                      &alpha, desc, pDevTempHiddenGrad,
                                      &alpha, desc, pDevPrevHiddenGrad));
        }

        m_aInput2Hidden->BackPropagateOnGPU(pTime);

        // delete, data loader, reset algo, 등 구하기

        return TRUE;
    }

#endif

    int ResetResult() {
        m_aInput2Hidden->ResetResult();
        m_aHidden2Hidden->ResetResult();
        m_aTempHidden->ResetResult();
        m_aPrevActivate->ResetResult();
        m_aPostActivate->ResetResult();
        m_aHidden2Output->ResetResult();
    }

    int ResetGradient() {
        m_aInput2Hidden->ResetGradient();
        m_aHidden2Hidden->ResetGradient();
        m_aTempHidden->ResetGradient();
        m_aPrevActivate->ResetGradient();
        m_aPostActivate->ResetGradient();
        m_aHidden2Output->ResetGradient();
    }

 // if __CUDNN__
};


#endif  // RECURRENT_H_
