#ifdef __CUDNN__

#include "L2Normalize.hpp"

// template class ConcatenateChannelWise<int>;
template class L2Normalize<float>;
// template class ConcatenateChannelWise<double>;

__global__ void Normalize_kernel(int sizeOfInputImg, int batchSize, float *input, float* norm2List, float* result) {
    
    int batchNum = blockIdx.x * blockDim.x + threadIdx.x;

    if(batchNum < batchSize)
    {
              
        norm2List[batchNum] = 0; 

        
        
        for (int elementIndex = 0; elementIndex < sizeOfInputImg; elementIndex++)
        {
            // printf("eidx: %d\n", elementIndex);
            norm2List[batchNum] += (input[batchNum * sizeOfInputImg + elementIndex] * input[batchNum * sizeOfInputImg + elementIndex]);

        }

        norm2List[batchNum] = sqrt(norm2List[batchNum]);

        for (int elementIndex = 0; elementIndex < sizeOfInputImg; elementIndex++)
        {
            // printf("norm2List[%d] = %f\n", batchNum, norm2List[batchNum]);

            result[batchNum * sizeOfInputImg + elementIndex] = (input[batchNum * sizeOfInputImg + elementIndex] / norm2List[batchNum]);

        }
    }

       
            
                        
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


template<typename DTYPE> int L2Normalize<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    int noBlock = 3, threadsPerBlock = 128;

    Tensor<DTYPE> *result = this->GetResult();
    Tensor<DTYPE> *input  = NULL;

    int timesize    = result->GetTimeSize();
    int batchsize   = result->GetBatchSize();
    int channelsize = result->GetChannelSize();
    int rowsize     = result->GetRowSize();
    int colsize     = result->GetColSize();

    Shape *resultTenShape = result->GetShape();

    int sizeOfPlane     = rowsize * colsize;
    int sizeOfInputImg  = 0;

    DTYPE *result_gpu = result->GetGPUData();
    DTYPE *input_gpu  = NULL;

    int inputChannelSize = 0;
    int idOfDevice = result->GetIdOfDevice();

    float test= 0.f;

    DTYPE* norm2ListGPUData;

    // std::cout << "L2 forward" << std::endl;

    Tensor<DTYPE>* testnorm2List = NULL;

    if(this->norm2ListGPU_);
        delete this->norm2ListGPU_;

    this->norm2ListGPU_ = Tensor<DTYPE>::Zeros(1, 1, 1, 1, batchsize);

    // testnorm2List = Tensor<DTYPE>::Zeros(1, 1, 1, 1, batchsize);


    // std::cout << "norm2 tensor: " << (*testnorm2List)[0] << std::endl;
    // for (int batchIndex = 0; batchIndex < batchsize; batchIndex++)
        // this->norm2ListGPU_[batchIndex] = 0;
    // std::cout << "testnorm2LIst zero index: " << (*testnorm2List)[0] << std::endl;
    // std::cout << "norm2 before get value: " << (*this->norm2ListGPU_)[0] << std::endl;


    if (this->norm2ListGPU_->GetDevice() != GPU)
        this->norm2ListGPU_->SetDeviceGPU(idOfDevice);

    // std::cout << "id: " << idOfDevice << std::endl;

    // testnorm2List->SetDeviceGPU(idOfDevice);
    // (*testnorm2List)[0]++;
    
    

    norm2ListGPUData = this->norm2ListGPU_->GetGPUData();
    // norm2ListGPUData = testnorm2List->GetGPUData();
    

    input            = this->GetInput()[0]->GetResult();
    input_gpu        = input->GetGPUData();
    // resultGPU        = result->GetGPUData();
    inputChannelSize = input->GetChannelSize();
    sizeOfInputImg   = inputChannelSize * sizeOfPlane;
    GetKernelParameters(batchsize, &noBlock, &threadsPerBlock);

    // std::cout << input->GetShape() << std::endl;
    // std::cout << "noblock: " << noBlock << " nothread: " << threadsPerBlock << std::endl;;
    

    // std::cout << "norm2 before getsum: " << (*this->norm2ListGPU_)[0] << std::endl;
    
    // std::cout << "norm2 before get value: " << (*this->norm2ListGPU_)[0] << std::endl;
    
    Normalize_kernel << < noBlock, threadsPerBlock >> > (sizeOfInputImg, batchsize, input_gpu, norm2ListGPUData, result_gpu);
    
    cudaDeviceSynchronize();
    

    // std::cout << "end of normalization" << std::endl;

    this->norm2ListGPU_->SetDeviceCPU();
    // testnorm2List->SetDeviceCPU();

    // (*this->norm2ListGPU_)[0] = 1.f;

    // std::cout << "normListgpu size: " << this->norm2ListGPU_->GetShape() << std::endl;
    // for(int i = 0; i < batchsize; i++)
    //     std::cout << "real norm 2 [" << i << "]: " << (*this->norm2ListGPU_)[i] << std::endl;

    // std::cout << "normalized value" << std::endl;
    
    // std::cout << "after normlization" << std::endl;

    /*
    
    for(int h = 0; h < batchsize; h++)
    {
        std::cout << "input vallue [" << h << "]: [" << std::endl;;
        for(int i = 0; i < sizeOfInputImg; i++)
            std::cout << (*input)[h * sizeOfInputImg + i] << std::endl;

        std::cout << "]" << std::endl;

        std::cout << "real norm 2 [" << h << "]: " << (*this->norm2ListGPU_)[h] << std::endl;


        std::cout << "result vallue [" << h << "]: [" << std::endl;;
        for(int i = 0; i < sizeOfInputImg; i++)
            std::cout << (*result)[h * sizeOfInputImg + i] << std::endl;

        std::cout << "]" << std::endl;
    }
    */
    
    // std::cout << "norm2 before sqrt: " << (norm2ListGPUData)[0] << std::endl;
    
    
    // std::cout << "norm2: " << (*testnorm2List)[0] << std::endl;


    // if (this->norm2ListGPU_->GetDevice() != GPU)
    // this->norm2ListGPU_->SetDeviceGPU(idOfDevice);

    // norm2ListGPUData = this->norm2ListGPU_->GetGPUData();



    return TRUE;
}




__global__ void L2Backprop_kernel(int sizeOfInputImg, int batchSize, float *thisDelta, float *inputDelta, float* norm2List, float* result) {
    
    int batchNum = blockIdx.x * blockDim.x + threadIdx.x;

    if(batchNum < batchSize)
    {
              
        // printf("bathnum in kernel: %d\n", batchNum);
        // printf("size of sizeofINput: %d\n", sizeOfInputImg);

        float sumOfDelta = 0;
        
        for (int elementIndex = 0; elementIndex < sizeOfInputImg; elementIndex++)
        {
            sumOfDelta += (thisDelta[batchNum * sizeOfInputImg + elementIndex] * result[batchNum * sizeOfInputImg + elementIndex]);
        }



        for (int elementIndex = 0; elementIndex < sizeOfInputImg; elementIndex++)
        {
            // printf("norm2List[%d] = %f\n", batchNum, norm2List[batchNum]);

            inputDelta[batchNum * sizeOfInputImg + elementIndex] += (thisDelta[batchNum * sizeOfInputImg + elementIndex] - (result[batchNum * sizeOfInputImg + elementIndex] * sumOfDelta)) / norm2List[batchNum];

        }
    }

       
            
                        
}

template<typename DTYPE> int L2Normalize<DTYPE>::BackPropagateOnGPU(int pTime) {

    // std::cout << "backprooppp!!" << std::endl;

    int noBlock = 3, threadsPerBlock = 64;
    Tensor<DTYPE> *this_delta  = this->GetGradient();
    Tensor<DTYPE> *inputDelta = this->GetInput()[0]->GetDelta();
    Tensor<DTYPE> *result = this->GetResult();
    Tensor<DTYPE> *input = this->GetInput()[0]->GetResult();
    
    DTYPE *result_gpu = result->GetGPUData();

    int timesize    = this_delta->GetTimeSize();
    int batchsize   = this_delta->GetBatchSize();
    int channelsize = this_delta->GetChannelSize();
    int rowsize     = this_delta->GetRowSize();
    int colsize     = this_delta->GetColSize();

    
    Shape *resultTenShape = this_delta->GetShape();

    int sizeOfPlane     = rowsize * colsize;
    int sizeOfResultImg = channelsize * sizeOfPlane;
    int sizeOfInputImg  = 0;

    DTYPE *thisDeltaGPU      = this_delta->GetGPUData();
    DTYPE *input_delta_gpu = inputDelta->GetGPUData();

    int idOfDevice = result->GetIdOfDevice();

    DTYPE* norm2ListGPUData;


    // std::cout << "back nromlistgpu: " << this->norm2ListGPU_->GetShape() << std::endl;
    // std::cout << "norm2gpu test" << (*this->norm2ListGPU_)[0] << std::endl; 
    // std::cout << "gpu id trial: " << idOfDevice << std::endl;
    // std::cout << "thisDeltagpu: " << this_delta->GetIdOfDevice() << std::endl;

    // if (sumOfDelta->GetDevice() != GPU)
    this->norm2ListGPU_->SetDeviceGPU(idOfDevice);

    
    norm2ListGPUData = this->norm2ListGPU_->GetGPUData();

    GetKernelParameters(batchsize, &noBlock, &threadsPerBlock);



    L2Backprop_kernel << < noBlock, threadsPerBlock >> > (sizeOfResultImg, batchsize, thisDeltaGPU, input_delta_gpu, norm2ListGPUData, result_gpu);

    cudaDeviceSynchronize();

    // std::cout << "l2 bp endss" << std::endl;

    this->norm2ListGPU_->SetDeviceCPU();
    // std::cout << "normListGPU[0]" << (*this->norm2ListGPU_)[0] << std::endl;

    /*
    for(int h = 5; h < 6; h++)
    {
        std::cout << "input vallue [" << h << "]: [" << std::endl;;
        for(int i = 0; i < sizeOfResultImg; i++)
            std::cout << (*input)[h * sizeOfResultImg + i] << std::endl;

        std::cout << "]" << std::endl;

        std::cout << "real norm 2 [" << h << "]: " << (*this->norm2ListGPU_)[h] << std::endl;

        std::cout << "result vallue [" << h << "]: [" << std::endl;;
        for(int i = 0; i < sizeOfResultImg; i++)
            std::cout << (*result)[h * sizeOfResultImg + i] << std::endl;

        std::cout << "]" << std::endl;

        std::cout << "thisDelta [" << h << "]: [" << std::endl;;
        for(int i = 0; i < sizeOfResultImg; i++)
            std::cout << (*this_delta)[h * sizeOfResultImg + i] << std::endl;

        std::cout << "]" << std::endl;
        
        std::cout << "inputDelta [" << h << "]: [" << std::endl;;
        for(int i = 0; i < sizeOfResultImg; i++)
            std::cout << (*inputDelta)[h * sizeOfResultImg + i] << std::endl;

        std::cout << "]" << std::endl;

    }
    */
    
   
    
    return TRUE;
}

#endif  // ifdef __CUDNN__
