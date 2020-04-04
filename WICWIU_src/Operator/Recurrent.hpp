#ifndef RECURRENT_H_
#define RECURRENT_H_    value

#include "../Operator.hpp"
#include "MatMul.hpp"
#include "Add.hpp"
#include "Tanh.hpp"
#include "Tensorholder.hpp"

#define TIME     0
#define BATCH    1

template<typename DTYPE> class Recurrent : public Operator<DTYPE>{
private:
    Operator<DTYPE> *m_aInput2Hidden;
    Operator<DTYPE> *m_aHidden2Hidden;
    Operator<DTYPE> *m_aTempHidden;
    Operator<DTYPE> *m_aPrevActivate;
    Operator<DTYPE> *ApplyActivation;
    Operator<DTYPE> *AddBias;

public:
    Recurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, rBias) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, rBias);
    }

    //pName때문에 Operator 생성자 호출이 안되는듯!!!!   숫자 4로해도 되는건가?
    Recurrent(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias, std::string pName) : Operator<DTYPE>(4, pInput, pWeightIH, pWeightHH, rBias, pName) {
        #if __DEBUG__
        std::cout << "Recurrent::Recurrent(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, pWeightIH, pWeightHH, rBias);
    }

    ~Recurrent() {
        #if __DEBUG__
        std::cout << "Recurrent::~Recurrent()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *pWeightIH, Operator<DTYPE> *pWeightHH, Operator<DTYPE> *rBias) {

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

        //For AnalyzeGraph
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
      void InitializeAttributeForGPU() {
          m_aInput2Hidden->SetDeviceGPU();
          m_aInput2Hidden->SetCudnnHandle(this->GetCudnnHandle());

          m_aTempHidden->SetDeviceGPU();
          m_aTempHidden->SetCudnnHandle(this->GetCudnnHandle());

          m_aHidden2Hidden->SetDeviceGPU();
          m_aHidden2Hidden->SetCudnnHandle(this->GetCudnnHandle());

          m_aPrevActivate->SetDeviceGPU();
          m_aPrevActivate->SetCudnnHandle(this->GetCudnnHandle());

          AddBias->SetDeviceGPU();
          AddBias->SetCudnnHandle(this->GetCudnnHandle());

          ApplyActivation->SetDeviceGPU();
          ApplyActivation->SetCudnnHandle(this->GetCudnnHandle());
      }

#endif  // if __CUDNN__

    //이거 해줘야되나?
    void Delete() {}


    int  ForwardPropagate(int pTime = 0) {

        #if __RNNDEBUG__
        std::cout <<pTime<<"번쨰 Recurrent forward 호출" << '\n';
        #endif  // __RNNDEBUG__

        m_aInput2Hidden->ForwardPropagate(pTime);

        if (pTime != 0) {
            Tensor<DTYPE> *prevHidden = ApplyActivation->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            int colSize        = prevHidden->GetColSize();
            Shape *HiddenShape = prevHidden->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*tempHidden)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] = (*prevHidden)[Index5D(HiddenShape, pTime - 1, 0, 0, 0, i)];
            }

            m_aHidden2Hidden->ForwardPropagate(pTime);
        }
        m_aPrevActivate->ForwardPropagate(pTime);

        AddBias->ForwardPropagate(pTime);

        ApplyActivation->ForwardPropagate(pTime);


        Tensor<DTYPE> *_result = ApplyActivation->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        Shape *ResultShape = result->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*result)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }


        return TRUE;
    }

    int BackPropagate(int pTime = 0) {

      #if __RNNDEBUG__
      std::cout <<pTime<<"번쨰 Recurrent BackPropagate 호출" << '\n';
      #endif  // __RNNDEBUG__

        Tensor<DTYPE> *_grad = ApplyActivation->GetGradient();
        Tensor<DTYPE> *grad  = this->GetGradient();

        int colSize        = grad->GetColSize();
        int timeSize       = grad->GetTimeSize();
        Shape *ResultShape = grad->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*_grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }

        if (pTime != timeSize-1) {
            m_aHidden2Hidden->BackPropagate(pTime+1);

            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
            Tensor<DTYPE> *prevHiddenGrad = ApplyActivation->GetGradient();

            int colSize        = tempHiddenGrad->GetColSize();
            Shape *HiddenShape = tempHiddenGrad->GetShape();

            for (int i = 0; i < colSize; i++) {
                (*prevHiddenGrad)[Index5D(HiddenShape, pTime, 0, 0, 0, i)] += (*tempHiddenGrad)[Index5D(HiddenShape, pTime+1, 0, 0, 0, i)];
            }
        }
        ApplyActivation->BackPropagate(pTime);

        AddBias->BackPropagate(pTime);

        m_aPrevActivate->BackPropagate(pTime);

        m_aInput2Hidden->BackPropagate(pTime);

        return TRUE;
    }


#if __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {
        int alpha = 1.f;

        cudnnTensorDescriptor_t desc = NULL;

        m_aInput2Hidden->ForwardPropagateOnGPU(pTime);

        if (pTime != 0) {
            Tensor<DTYPE> *prevHidden = ApplyActivation->GetResult();
            Tensor<DTYPE> *tempHidden = m_aTempHidden->GetResult();

            //DTYPE *pDevPrevHidden = prevHidden->GetDeviceData(pTime - 1);
            //DTYPE *pDevTempHidden = tempHidden->GetDeviceData(pTime);

            DTYPE *pDevPrevHidden = prevHidden->GetGPUData(pTime - 1);
            DTYPE *pDevTempHidden = tempHidden->GetGPUData(pTime);

            desc = prevHidden->GetDescriptor();

            checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                      &alpha, desc, pDevPrevHidden,
                                      &alpha, desc, pDevTempHidden));

            m_aHidden2Hidden->ForwardPropagateOnGPU(pTime);
        }

        m_aPrevActivate->ForwardPropagateOnGPU(pTime);

        AddBias->ForwardPropagateOnGPU(pTime);

        ApplyActivation->ForwardPropagateOnGPU(pTime);

        Tensor<DTYPE> *_result = ApplyActivation->GetResult();
        Tensor<DTYPE> *result  = this->GetResult();

        int colSize        = result->GetColSize();
        Shape *ResultShape = result->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*result)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*_result)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }

        return TRUE;
    }


    int BackPropagateOnGPU(int pTime = 0) {
        int alpha = 1.f;

        cudnnTensorDescriptor_t desc = NULL;

        Tensor<DTYPE> *_grad = ApplyActivation->GetGradient();
        Tensor<DTYPE> *grad  = this->GetGradient();

        //이거는.... 사용안하는거 같은데....
        //DTYPE *_pDevGrad = _grad->GetDeviceData(pTime);
        //DTYPE *pDevGrad  = grad->GetDeviceData(pTime);

        int colSize        = grad->GetColSize();
        int timeSize       = grad->GetTimeSize();
        Shape *ResultShape = grad->GetShape();

        for (int i = 0; i < colSize; i++) {
            (*_grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)] = (*grad)[Index5D(ResultShape, pTime, 0, 0, 0, i)];
        }

        if (pTime != timeSize-1) {
            m_aHidden2Hidden->BackPropagateOnGPU(pTime);

            Tensor<DTYPE> *tempHiddenGrad = m_aTempHidden->GetGradient();
            Tensor<DTYPE> *prevHiddenGrad = ApplyActivation->GetGradient();

            //DTYPE *pDevTempHiddenGrad = tempHiddenGrad->GetDeviceData(pTime + 1);
            //DTYPE *pDevPrevHiddenGrad = prevHiddenGrad->GetDeviceData(pTime);

            DTYPE *pDevTempHiddenGrad = tempHiddenGrad->GetGPUData(pTime + 1);
            DTYPE *pDevPrevHiddenGrad = prevHiddenGrad->GetGPUData(pTime);

            desc = tempHiddenGrad->GetDescriptor();

            checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                                      &alpha, desc, pDevTempHiddenGrad,
                                      &alpha, desc, pDevPrevHiddenGrad));
        }
        ApplyActivation->BackPropagateOnGPU(pTime);

        AddBias->BackPropagateOnGPU(pTime);

        m_aPrevActivate->BackPropagateOnGPU(pTime);

        m_aInput2Hidden->BackPropagateOnGPU(pTime);

        // delete, data loader, reset algo, 등 구하기

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
    }

    int ResetGradient() {
        m_aInput2Hidden->ResetGradient();
        m_aHidden2Hidden->ResetGradient();
        m_aTempHidden->ResetGradient();
        m_aPrevActivate->ResetGradient();
        ApplyActivation->ResetGradient();
        AddBias->ResetGradient();
    }


};


#endif  // RECURRENT_H_
