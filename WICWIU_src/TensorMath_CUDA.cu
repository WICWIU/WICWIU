#ifdef __CUDNN__

#include "TensorMath.cpp"

template void ArgmaxOnGPU<float>(Tensor<float> *, Tensor<float> *, int);

__global__ void MemcpyDevInt2DevFloat(float *farray, int *iarray, int total) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < total) {
        farray[index] = (float)iarray[index];
    }
}


template <typename DTYPE> void ArgmaxOnGPU(Tensor<DTYPE> *resultTensor, Tensor<DTYPE> *inputTensor, int dim) {
    cudnnHandle_t& handle = resultTensor->GetCudnnHandle();

    DTYPE *inputArray = inputTensor->GetGPUData();  // inferred tensor
    DTYPE *resultArray = resultTensor->GetGPUData();

    int timesize = inputTensor->GetTimeSize();
    int batchsize = inputTensor->GetBatchSize();
    int channelsize = inputTensor->GetChannelSize();
    int rowsize = inputTensor->GetRowSize();
    int colsize = inputTensor->GetColSize();

    int resultDim[5] = {timesize, batchsize, channelsize, rowsize, colsize};
    resultDim[dim] = 1;

    float alpha = 1.0;
    float beta = 0.F;

    cudnnTensorDescriptor_t aDesc;
    cudnnTensorDescriptor_t cDesc;

    checkCUDNN(cudnnCreateTensorDescriptor(&aDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&cDesc));

    checkCUDNN(cudnnSetTensor4dDescriptor(aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchsize, channelsize, rowsize, colsize));
    checkCUDNN(cudnnSetTensor4dDescriptor(cDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, resultDim[1], resultDim[2], resultDim[3], resultDim[4]));

    cudnnReduceTensorDescriptor_t reduceTenDesc;

    checkCUDNN(cudnnCreateReduceTensorDescriptor(&reduceTenDesc));
    checkCUDNN(cudnnSetReduceTensorDescriptor(reduceTenDesc, CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES, CUDNN_32BIT_INDICES));

    size_t workspaceSize;
    size_t indicesSize;

    checkCUDNN(cudnnGetReductionWorkspaceSize(handle, reduceTenDesc, aDesc, cDesc, &workspaceSize));
    checkCUDNN(cudnnGetReductionIndicesSize(handle, reduceTenDesc, aDesc, cDesc, &indicesSize));

    void* workspace;
    void* indices;

    checkCudaErrors(cudaMalloc(&workspace, workspaceSize));
    checkCudaErrors(cudaMalloc(&indices, indicesSize));
    checkCUDNN(cudnnReduceTensor(handle, reduceTenDesc, indices, indicesSize, workspace, workspaceSize, &alpha, aDesc, inputArray, &beta, cDesc, resultArray));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCUDNN(cudnnDestroyReduceTensorDescriptor(reduceTenDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(aDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(cDesc));

    int totalThread = indicesSize/sizeof(int);
    int noBlock, threadsPerBlock;
    GetKernelParameters(totalThread, &noBlock, &threadsPerBlock);

    MemcpyDevInt2DevFloat<<<noBlock, threadsPerBlock>>>(resultArray, (int *)indices, totalThread);

    checkCudaErrors(cudaFree(workspace));
    checkCudaErrors(cudaFree(indices));

}

#endif // __CUDNN__
