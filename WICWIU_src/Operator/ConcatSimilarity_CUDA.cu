#ifdef __CUDNN__


#include "MatMul.hpp"
#include "Tanh.hpp"
#include "Transpose.hpp"
#include "Tensorholder.hpp"
#include "BroadMatMul.hpp"
#include "ConcatSimilarity.hpp"

template class ConcatSimilarity<float>;


//UH temp result
__global__ void Add_ForwardPropagate_kernel(float *input1, float *input2, float *result, int timesize, int batchsize, int colsize, int weightDim)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim; idx += blockDim.x * gridDim.x) {

        int batch = idx / (timesize * colsize);         
        int inputIndex = idx%colsize + (batch * colsize);    

        result[idx] += input1[idx] * input2[inputIndex];

    }

}

__global__ void Dot_ForwardPropagate_kernel(float *input, float *weight, float *result, int batchsize, int rowsize, int colsize)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rowsize * colsize; idx += blockDim.x * gridDim.x) {

        for(int ba =0; ba < batchsize; ba++){
          int resultIndex = idx / colsize + ba * rowsize;
          int inputIndex = idx % colsize + ba * colsize;

          result[resultIndex] += weight[idx] * input[inputIndex];
        }
    }

}


template<typename DTYPE> int ConcatSimilarity<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    int noBlock = 3, threadsPerBlock = 128;

    Tensor<DTYPE> *key = this->GetInput()[0]->GetResult();
    int keytimesize = key->GetTimeSize();

    if(pTime == 0){
        for(int ti = 0; ti < keytimesize; ti++)    m_aKeyMatMul->ForwardPropagateOnGPU(ti);

        m_aKeyMatMulTranspose->ForwardPropagateOnGPU();

        return TRUE;       
    }


    // ---------------------  Wa * Si-1    Decoder 이전 time hidden   ------------------------           
    Tensor<DTYPE> *query  = this->GetInput()[4]->GetResult();
    Tensor<DTYPE> *weight  = this->GetInput()[2]->GetResult();      
    Tensor<DTYPE> *temp    = m_aTemp->GetResult();

    int batchsize        = query->GetBatchSize();
    int weightrowsize    = weight->GetRowSize();
    int weightcolsize    = weight->GetColSize();

    DTYPE *m_pDevInput  = query->GetGPUData(pTime-1);
    DTYPE *m_pDevWeight  = weight->GetGPUData();
    DTYPE *m_pDevOutput = temp->GetGPUData(pTime);


    Dot_ForwardPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevInput, m_pDevWeight, m_pDevOutput, batchsize, weightrowsize, weightcolsize);

    // add
    Tensor<DTYPE> *UH = m_aKeyMatMulTranspose->GetResult();   
    Tensor<DTYPE> *addResult = m_aPrevActivate->GetResult();    

    DTYPE *m_pDevInput1  = UH->GetGPUData();        //transpose를 해서 time = 0
    DTYPE *m_pDevInput2  = temp->GetGPUData(pTime);
    m_pDevOutput  = addResult->GetGPUData(pTime);

    int parameterDim = UH->GetCapacity();
    batchsize = UH->GetBatchSize();
    int colsize = UH->GetColSize();

    GetKernelParameters(parameterDim, &noBlock, &threadsPerBlock);

    Add_ForwardPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevInput1, m_pDevInput2, m_pDevOutput, keytimesize, batchsize, colsize, parameterDim);

    ApplyActivation->ForwardPropagateOnGPU(pTime);        //tanh
    m_aMatMul->ForwardPropagateOnGPU(pTime);              //matmul


    Tensor<DTYPE> *_result = m_aMatMul->GetResult();
    Tensor<DTYPE> *result  = this->GetResult();

    // std::cout<<"복사 전 값"<<'\n';
    // std::cout<<_result<<'\n';

    DTYPE *_pDevResult = _result->GetGPUData(pTime);
    DTYPE *pDevresult = result->GetGPUData(pTime);

    cudnnTensorDescriptor_t desc = result->GetDescriptor();

    DTYPE m_alpha = 1;
    DTYPE m_beta  = 0;

    checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                              &m_alpha, desc, _pDevResult,
                              &m_beta, desc, pDevresult));


    return TRUE;
}


__global__ void Add_BackPropagate_kernel(float *input1, float *input2, float *result, int timesize, int batchsize, int colsize, int weightDim)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < weightDim; idx += blockDim.x * gridDim.x) {

        int batch = idx / (timesize * colsize);
        int inputIndex = idx%colsize + (batch * colsize);

        input1[idx] += result[idx];
        input2[inputIndex] += result[idx];

    }

}


__global__ void Dot_BackPropagate_kernel(float *input, float *inputGradient, float *weight, float *weightGradient, float *resultGradient, int batchsize, int rowsize, int colsize)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rowsize * colsize; idx += blockDim.x * gridDim.x) {

        for(int ba =0; ba < batchsize; ba++){
          int resultIndex = idx / colsize + ba * rowsize;
          int inputIndex = idx % colsize + ba * colsize;

          weightGradient[idx] += resultGradient[resultIndex] * input[inputIndex];    
          inputGradient[inputIndex] += resultGradient[resultIndex] * weight[idx];

        }
    }

}


template<typename DTYPE> int ConcatSimilarity<DTYPE>::BackPropagateOnGPU(int pTime) {
    int noBlock = 3, threadsPerBlock = 128;

    Tensor<DTYPE> *key = this->GetInput()[0]->GetResult();
    int keytimesize = key->GetTimeSize();

    Tensor<DTYPE> *_result = m_aMatMul->GetGradient();
    Tensor<DTYPE> *result  = this->GetGradient();

    DTYPE *_pDevResult = _result->GetGPUData(pTime);
    DTYPE *pDevresult = result->GetGPUData(pTime);

    cudnnTensorDescriptor_t desc = result->GetDescriptor();

    DTYPE m_alpha = 1;
    DTYPE m_beta  = 0;

    checkCUDNN(cudnnAddTensor(this->GetCudnnHandle(),
                              &m_alpha, desc, pDevresult,
                              &m_beta, desc, _pDevResult));

    m_aMatMul->BackPropagateOnGPU(pTime);
    ApplyActivation->BackPropagateOnGPU(pTime);

    Tensor<DTYPE> *UH = m_aKeyMatMulTranspose->GetGradient();      
    Tensor<DTYPE> *addResult = m_aPrevActivate->GetGradient();     
    Tensor<DTYPE> *temp    = m_aTemp->GetGradient();

    DTYPE *m_pDevInput1  = UH->GetGPUData();       
    DTYPE *m_pDevInput2  = temp->GetGPUData(pTime);
    DTYPE *m_pDevOutput  = addResult->GetGPUData(pTime);

    int parameterDim = UH->GetCapacity() / UH->GetTimeSize();
    int batchsize = UH->GetBatchSize();
    int colsize = UH->GetColSize();

    Add_BackPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevInput1, m_pDevInput2, m_pDevOutput, keytimesize, batchsize, colsize, parameterDim);

    if(pTime == 0){
        m_aKeyMatMulTranspose->BackPropagateOnGPU();

        for(int ti = keytimesize-1; ti >=0; ti--)    m_aKeyMatMul->BackPropagateOnGPU(ti);
        return TRUE;       
    }

    Tensor<DTYPE> *query  = this->GetInput()[4]->GetResult();
    Tensor<DTYPE> *queryGradient  = this->GetInput()[4]->GetGradient();
    Tensor<DTYPE> *weight  = this->GetInput()[2]->GetResult();
    Tensor<DTYPE> *weightGradient  = this->GetInput()[2]->GetGradient();     

    batchsize        = query->GetBatchSize();
    int weightrowsize    = weight->GetRowSize();
    int weightcolsize    = weight->GetColSize();

    DTYPE *m_pDevInput           = query->GetGPUData(pTime-1);
    DTYPE *m_pDevInputGradient   = queryGradient->GetGPUData(pTime-1);
    DTYPE *m_pDevWeight          = weight->GetGPUData();
    DTYPE *m_pDevWeightGradient  = weightGradient->GetGPUData();
    m_pDevOutput          = temp->GetGPUData(pTime);


    Dot_BackPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevInput, m_pDevInputGradient, m_pDevWeight, m_pDevWeightGradient, m_pDevOutput, batchsize, weightrowsize, weightcolsize);      


    return TRUE;
}

#endif  // __CUDNN__
