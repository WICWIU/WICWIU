#ifndef LRELU_H_
#define LRELU_H_    value

#include "../Operator.h"

template<typename DTYPE> class LRelu : public Operator<DTYPE>{
private:
      float m_negativeSlope;

  #ifdef __CUDNN__
      cudnnTensorDescriptor_t m_aInputTensorDesc, m_aOutputTensorDesc, m_aDeltaDesc, m_aInputDeltaDesc;
      cudnnActivationDescriptor_t actDesc;
      DTYPE *m_pDevInput, *m_pDevOutput, *m_pDevInputDelta, *m_pDevDelta;

      float m_alpha;
      float m_beta;
      double m_coef;

  #endif  // __CUDNN__


public:
    LRelu(Operator<DTYPE> *pInput, float negativeSlope) : Operator<DTYPE>(pInput) {
        #ifdef __DEBUG__
        std::cout << "LRelu::LRelu(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, negativeSlope);
    }

    LRelu(Operator<DTYPE> *pInput, float negativeSlope, std::string pName) : Operator<DTYPE>(pInput, pName) {
        #ifdef __DEBUG__
        std::cout << "LRelu::LRelu(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, negativeSlope);

    }

    ~LRelu() {
        #ifdef __DEBUG__
        std::cout << "LRelu::~LRelu()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }

    int Alloc(Operator<DTYPE> *pInput, float negativeSlope) {
        #ifdef __DEBUG__
        std::cout << "LRelu::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        m_negativeSlope = negativeSlope;
        //std::cout << m_negativeSlope << '\n';

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));
        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

#ifdef __CUDNN__
    void InitializeAttributeForGPU(unsigned int idOfDevice) {
        Operator<DTYPE> *pInput = this->GetInput()[0];

        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = pInput->GetResult()->GetRowSize();
        int colsize     = pInput->GetResult()->GetColSize();

        int inputCapacity  = pInput->GetResult()->GetCapacity();
        int outputCapacity = this->GetResult()->GetCapacity();

        m_alpha = 1.f;
        m_beta  = 0.f;
        m_coef  = m_negativeSlope;

        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aOutputTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aDeltaDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&m_aInputDeltaDesc));
        checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));

        checkCUDNN(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, m_coef));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aOutputTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));

        checkCUDNN(cudnnSetTensor4dDescriptor(m_aInputDeltaDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              batchsize, channelsize, rowsize, colsize));
    }

#endif  // if __CUDNN__


    void Delete() {
      /*if (m_negativeSlope) {
          delete m_negativeSlope;
          m_negativeSlope = NULL;
      }*/
      //delete하는게 맞는지?
#ifdef __CUDNN__

        if (m_aInputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aInputTensorDesc));
        m_aInputTensorDesc = NULL;

        if (m_aOutputTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aOutputTensorDesc));
        m_aOutputTensorDesc = NULL;

        if (m_aDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aDeltaDesc));
        m_aDeltaDesc = NULL;

        if (m_aInputDeltaDesc) checkCUDNN(cudnnDestroyTensorDescriptor(m_aInputDeltaDesc));
        m_aInputDeltaDesc = NULL;

        if (actDesc) checkCUDNN(cudnnDestroyActivationDescriptor(actDesc));
        actDesc = NULL;

        checkCudaErrors(cudaThreadSynchronize());
#endif  // if __CUDNN__
    }

    int ForwardPropagate(int pTime = 0, int pThreadNum = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        int ti = pTime;
        //int numOfThread = this->GetNumOfThread();

		    /*
        for (int ba = pThreadNum; ba < batchsize; ba += numOfThread) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {

                        (*result)[Index5D(resultTenShape, ti, ba, ch, ro, co)]
                            = this->MAX((*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)], 0.f) + m_negativeSlope*this->MIN(0.f, (*input)[Index5D(resultTenShape, ti, ba, ch, ro, co)]);
                    }
                }
            }
        }
		    */

		    /* 	improved version 1
        for (int ba = pThreadNum; ba < batchsize; ba += numOfThread) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
            						int idx = Index5D(resultTenShape, ti, ba, ch, ro, co);
            						if((*input)[idx] >= 0.f)
            							(*result)[idx] = (*input)[idx];
            						else
            							(*result)[idx] = m_negativeSlope * (*input)[idx];
                    }
                }
            }
        }
		    */

		    /*  improved version 2
		    int idx = Index5D(resultTenShape, 0, 0, 0, 0, 0);
        for (int ba = pThreadNum; ba < batchsize; ba += numOfThread) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++, idx++) {
            						if((*input)[idx] >= 0.f)
            							(*result)[idx] = (*input)[idx];
            						else
            							(*result)[idx] = m_negativeSlope * (*input)[idx];
                    }
                }
            }
        }
		    */

		    //  improved version 3
		    int index = 0;
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {
                        index = Index5D(resultTenShape, ti, ba, ch, ro, co);
            						if((*input)[index] >= 0.f)
            							(*result)[index] = (*input)[index];
            						else
            							(*result)[index] = m_negativeSlope * (*input)[index];
                    }
                }
            }
        }


		    /*  improved version 4
    		int idx = Index5D(resultTenShape, 0, 0, 0, 0, 0);
    		DTYPE* ip = &(*input)[idx];
    		DTYPE* op = &(*result)[idx];

        for (int ba = pThreadNum; ba < batchsize; ba += numOfThread) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++, ip++, op++) {
            						if(*ip >= 0.f)
            							*op = *ip;
            						else
            							*op = m_negativeSlope * *ip;
                    }
                }
            }
        }
		    */

    		/* improved version 5
    		int index = Index5D(resultTenShape, 0, 0, 0, 0, 0); //initial index
    		DTYPE *input_ptr = &(*input)[index];
    		DTYPE *result_ptr = &(*result)[index];
    		DTYPE *input_limit = input_ptr + batchsize * channelsize * rowsize * colsize;
    		for(; input_ptr < input_limit; input_ptr++, result_ptr++){

      			if(*input_ptr >= 0.f)
      				*result_ptr = *input_ptr;
      			else
      				*result_ptr = m_negativeSlope * (*input_ptr);
    		}*/

        return TRUE;
    }

    int BackPropagate(int pTime = 0, int pThreadNum = 0) {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        int timesize    = result->GetTimeSize();
        int batchsize   = result->GetBatchSize();
        int channelsize = result->GetChannelSize();
        int rowsize     = result->GetRowSize();
        int colsize     = result->GetColSize();

        Shape *resultTenShape = result->GetShape();

        int ti = pTime;
        //int numOfThread = this->GetNumOfThread();

        int index = 0;
        for (int ba = 0; ba < batchsize; ba++) {
            for (int ch = 0; ch < channelsize; ch++) {
                for (int ro = 0; ro < rowsize; ro++) {
                    for (int co = 0; co < colsize; co++) {

                        index = Index5D(resultTenShape, ti, ba, ch, ro, co);

                        if ((*result)[index] > 0.f)
                            (*input_delta)[index] += (*this_delta)[index];
                        else
                            (*input_delta)[index] += m_negativeSlope * (*this_delta)[index];

                        //std::cout << "input_delta: " << (*input_delta)[Index5D(resultTenShape, ti, ba, ch, ro, co)] << std::endl;
                    }
                }
            }
        }

        /*improved version
        int index = Index5D(resultTenShape, 0, 0, 0, 0, 0); //initial index
        DTYPE *input_delta_ptr = &(*input_delta)[index];
        DTYPE *this_delta_ptr = &(*this_delta)[index];
        DTYPE *result_ptr = &(*result)[index];
        DTYPE *delta_limit = input_delta_ptr + batchsize * channelsize * rowsize * colsize;
        for(; input_delta_ptr < delta_limit; input_delta_ptr++, this_delta_ptr++, result_ptr++){

            if(*result_ptr > 0.f)
              *input_delta_ptr += *this_delta_ptr;
            else
              *input_delta_ptr += m_negativeSlope * (*this_delta_ptr);
        }
        */

        return TRUE;
    }


#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        m_pDevInput  = input->GetGPUData(pTime);
        m_pDevOutput = result->GetGPUData(pTime);

        checkCUDNN(cudnnActivationForward(this->GetCudnnHandle(), actDesc,
                                          &m_alpha, m_aInputTensorDesc, m_pDevInput,
                                          &m_beta, m_aOutputTensorDesc, m_pDevOutput));

        checkCudaErrors(cudaDeviceSynchronize());
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime = 0) {
        Tensor<DTYPE> *result      = this->GetResult();
        Tensor<DTYPE> *this_delta  = this->GetGradient();
        Tensor<DTYPE> *input       = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *input_delta = this->GetInput()[0]->GetDelta();

        m_pDevInput      = input->GetGPUData(pTime);
        m_pDevOutput     = result->GetGPUData(pTime);
        m_pDevDelta      = this_delta->GetGPUData(pTime);
        m_pDevInputDelta = input_delta->GetGPUData(pTime);

        checkCUDNN(cudnnActivationBackward(this->GetCudnnHandle(), actDesc, &m_alpha,
                                           m_aOutputTensorDesc, m_pDevOutput,
                                           m_aDeltaDesc, m_pDevDelta,
                                           m_aInputTensorDesc, m_pDevInput, &m_beta,
                                           m_aInputTensorDesc, m_pDevInputDelta));

        checkCudaErrors(cudaDeviceSynchronize());

        return TRUE;
    }

#endif  // if __CUDNN__
/*
#ifdef __CUDNN__
    int ForwardPropagateOnGPU(int pTime) {
        this->ForwardPropagate(pTime);
        return TRUE;
    }

    int BackPropagateOnGPU(int pTime) {
        this->BackPropagate(pTime);

        return TRUE;
    }

#endif  // __CUDNN__
*/

};


#endif  // LRELU_H_
