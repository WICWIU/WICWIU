#ifndef SWITCH_H_
#define SWITCH_H_ value

#include "../Operator.hpp"

template <typename DTYPE>
class Switch : public Operator<DTYPE>
{
private:
    int m_SwitchNumber;

public:
    Switch(Operator<DTYPE>* pInput0, Operator<DTYPE>* pInput1, int pLoadflag = TRUE)
        : Operator<DTYPE>(pInput0, pInput1, "NO_NAME", pLoadflag)
    {
#ifdef __DEBUG__
        std::cout << "Switch::Switch(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
#endif // __DEBUG__

        m_SwitchNumber = -1;

        this->Alloc(pInput0, pInput1);
    }

    Switch(Operator<DTYPE>* pInput0, Operator<DTYPE>* pInput1, std::string pName,
           int pLoadflag = TRUE)
        : Operator<DTYPE>(pInput0, pInput1, pName, pLoadflag)
    {
#ifdef __DEBUG__
        std::cout << "Switch::Switch(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
#endif // __DEBUG__

        m_SwitchNumber = -1;

        this->Alloc(pInput0, pInput1);
    }

    ~Switch()
    {
#ifdef __DEBUG__
        std::cout << "Switch::~Switch()" << '\n';
#endif // __DEBUG__

        Delete();
    }

    int GetSwitchNumber() { return m_SwitchNumber; }

    int SetSwitchNumber(int pSwitchNumber) { return m_SwitchNumber = pSwitchNumber; }

    int Alloc(Operator<DTYPE>* pInput0, Operator<DTYPE>* pInput1)
    {
#ifdef __DEBUG__
        std::cout << "Switch::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
#endif // __DEBUG__

        int timesize = pInput0->GetResult()->GetTimeSize();
        int batchsize = pInput0->GetResult()->GetBatchSize();
        int channelsize = pInput0->GetResult()->GetChannelSize();
        int rowsize = pInput0->GetResult()->GetRowSize();
        int colsize = pInput0->GetResult()->GetColSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    void Delete() {}

    int ForwardPropagate(int pTime = 0)
    {
        Tensor<DTYPE>* input0 = this->GetInput()[0]->GetResult();
        Tensor<DTYPE>* input1 = this->GetInput()[1]->GetResult();
        Tensor<DTYPE>* result = this->GetResult();

        int timesize = result->GetTimeSize();
        int batchsize = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize = result->GetRowSize();
        int colsize = result->GetColSize();

        Shape* resultTenShape = result->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++)
        {
            for (int ch = 0; ch < channelsize; ch++)
            {
                for (int ro = 0; ro < rowsize; ro++)
                {
                    for (int co = 0; co < colsize; co++)
                    {
                        if (m_SwitchNumber == 0)
                        {
                            (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)] =
                                (*input0)[Index5D(resultTenShape, ti, ba, ch, ro, co)];
                        }
                        else if (m_SwitchNumber == 1)
                        {
                            (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)] =
                                (*input1)[Index5D(resultTenShape, ti, ba, ch, ro, co)];
                        }
                        else
                        {
                            std::cout
                                << "Switch::ForwardPropagate() | Error : Invalid Switch Number."
                                << '\n';
                            exit(-1);
                        }
                    }
                }
            }
        }

        return TRUE;
    }

    int BackPropagate(int pTime = 0)
    {
        Tensor<DTYPE>* result = this->GetResult();
        Tensor<DTYPE>* this_delta = this->GetGradient();
        Tensor<DTYPE>* input0_delta = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE>* input1_delta = this->GetInput()[1]->GetDelta();

        int timesize = this_delta->GetTimeSize();
        int batchsize = this_delta->GetBatchSize();
        int channelsize = this_delta->GetChannelSize();
        int rowsize = this_delta->GetRowSize();
        int colsize = this_delta->GetColSize();

        Shape* resultTenShape = result->GetShape();

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++)
        {
            for (int ch = 0; ch < channelsize; ch++)
            {
                for (int ro = 0; ro < rowsize; ro++)
                {
                    for (int co = 0; co < colsize; co++)
                    {
                        if (m_SwitchNumber == 0)
                        {
                            (*input0_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)] +=
                                (*this_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)];
                        }
                        else if (m_SwitchNumber == 1)
                        {
                            (*input1_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)] +=
                                (*this_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)];
                        }
                        else
                        {
                            std::cout
                                << "Switch::BackwardPropagate() | Error : Invalid Switch Number."
                                << '\n';
                            exit(-1);
                        }
                    }
                }
            }
        }

        return TRUE;
    }

#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0)
    {
        Tensor<DTYPE>* input0 = this->GetInput()[0]->GetResult();
        Tensor<DTYPE>* input1 = this->GetInput()[1]->GetResult();
        Tensor<DTYPE>* result = this->GetResult();

        DTYPE* pDevInput0 = input0->GetGPUData(pTime);
        DTYPE* pDevInput1 = input1->GetGPUData(pTime);
        DTYPE* pDevOutput = result->GetGPUData(pTime);

        cudnnTensorDescriptor_t pDesc = input0->GetDescriptor();

        float alpha = 1.f;
        float beta = 0.f;

        if (m_SwitchNumber == 0)
        {
            checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(), &alpha, pDesc, pDevInput0, &beta,
                                      pDesc, pDevOutput));
        }
        else if (m_SwitchNumber == 1)
        {
            checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(), &alpha, pDesc, pDevInput1, &beta,
                                      pDesc, pDevOutput));
        }
        else
        {
            std::cout << "Switch::ForwardPropagate() | Error : Invalid Switch Number." << '\n';
            exit(-1);
        }

        // checkCudaErrors(cudaDeviceSynchronize());
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0)
    {
        Tensor<DTYPE>* this_delta = this->GetDelta();
        Tensor<DTYPE>* input0_delta = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE>* input1_delta = this->GetInput()[1]->GetDelta();

        DTYPE* pDevDelta = this_delta->GetGPUData();
        DTYPE* pDevInput0Delta = input0_delta->GetGPUData();
        DTYPE* pDevInput1Delta = input1_delta->GetGPUData();

        cudnnTensorDescriptor_t pDesc = this_delta->GetDescriptor();

        float alpha = 1.f;
        float beta = 0.f;

        if (m_SwitchNumber == 0)
        {
            checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(), &alpha, pDesc, pDevDelta, &beta,
                                      pDesc, pDevInput0Delta));
        }
        else if (m_SwitchNumber == 1)
        {
            checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(), &alpha, pDesc, pDevDelta, &beta,
                                      pDesc, pDevInput1Delta));
        }
        else
        {
            std::cout << "Switch::ForwardPropagate() | Error : Invalid Switch Number." << '\n';
            exit(-1);
        }

        // this->BackPropagate(pTime);

        return TRUE;
    }

#endif // if __CUDNN__
};

#endif // RELU_H_
