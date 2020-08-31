#include "../WICWIU_src/NeuralNetwork.hpp"

int main(int argc, char const* argv[])
{
    Tensorholder<float>* pWeight =
        new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, 3, 5, 0.0, 0.1), "x");
    Tensorholder<float>* input0 =
        new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, 1, 5, 0.0, 0.1), "label");

    std::cout << pWeight->GetResult()->GetShape() << '\n';
    std::cout << pWeight->GetResult() << '\n';

    std::cout << input0->GetResult()->GetShape() << '\n';
    std::cout << input0->GetResult() << '\n';

    Operator<float>* matmul = new MatMul<float>(pWeight, input0, "matmultest");

#ifdef __CUDNN__
    cudnnHandle_t m_cudnnHandle;
    cudnnCreate(&m_cudnnHandle);
    pWeight->SetDeviceGPU(m_cudnnHandle, 0);
    input0->SetDeviceGPU(m_cudnnHandle, 0);
    matmul->SetDeviceGPU(m_cudnnHandle, 0);
#endif // ifdef __CUDNN__

    std::cout << matmul->GetResult()->GetShape() << '\n';
    std::cout << matmul->GetResult() << '\n';
}
