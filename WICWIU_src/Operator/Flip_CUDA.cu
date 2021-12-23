#ifdef __CUDNN__

#include "Flip.hpp"

template class FlipTimeWise<float>;

__global__ void ForwardPropagate_kernel(float *pDevInput, float *pDevOutput, int capacityPerTime) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < capacityPerTime; idx += blockDim.x * gridDim.x) {
        pDevOutput[idx] = pDevInput[idx];
    }
}

template<typename DTYPE> int FlipTimeWise<DTYPE>::ForwardPropagateOnGPU(int pTime) {
        int noBlock = 3, threadsPerBlock = 128;

        if(pTime !=0)
          return TRUE;

        Tensor<DTYPE> *input  = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int timesize        = input->GetTimeSize();
        int capacityPerTime     = result->GetCapacity() / timesize;

        for(int ti = 0; ti < timesize; ti++){
            DTYPE *m_pDevInput  = input->GetGPUData(timesize - ti - 1);
            DTYPE *m_pDevOutput = result->GetGPUData(ti);

            ForwardPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevInput, m_pDevOutput, capacityPerTime);
        }


        return TRUE;
}


__global__ void BackPropagate_kernel(float *pDevInput, float *pDevOutput, int capacityPerTime) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < capacityPerTime; idx += blockDim.x * gridDim.x) {
      pDevInput[idx] = pDevOutput[idx];
  }
}


template<typename DTYPE> int FlipTimeWise<DTYPE>::BackPropagateOnGPU(int pTime) {
        int noBlock = 3, threadsPerBlock = 128;

        Tensor<DTYPE> *input  = this->GetInput()[0]->GetGradient();
        Tensor<DTYPE> *result = this->GetGradient();

        int timesize        = input->GetTimeSize();
        int capacityPerTime     = result->GetCapacity() / timesize;

        if(pTime != timesize-1)
          return TRUE;

        for(int ti = 0; ti < timesize; ti++){
            DTYPE *m_pDevInput  = input->GetGPUData(ti);
            DTYPE *m_pDevOutput = result->GetGPUData(timesize - ti - 1);

            BackPropagate_kernel << < noBlock, threadsPerBlock >> > (m_pDevInput, m_pDevOutput, capacityPerTime);
        }

        return TRUE;
}

#endif  // ifdef __CUDNN__
