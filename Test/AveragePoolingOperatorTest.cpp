#include "../WICWIU_src/NeuralNetwork.hpp"


int main(int argc, char const *argv[]) {
    Tensorholder<float> *input1 = new Tensorholder<float>(Tensor<float>::Constants(1, 1, 3, 4, 4, 2.0), "label");

    std::cout << input1->GetResult()->GetShape() << '\n';
    std::cout << input1->GetResult() << '\n';

    Operator<float> *avg = new AvaragePooling2D<float>(input1, 2, 2, 2, 2, 0, "AVG");

  #ifdef __CUDNN__
    cudnnHandle_t m_cudnnHandle;
    cudnnCreate(&m_cudnnHandle);
    input1->SetDeviceGPU(m_cudnnHandle, 0);
    avg->SetDeviceGPU(m_cudnnHandle, 0);
  #endif  // ifdef __CUDNN__

    std::cout << avg->GetResult()->GetShape() << '\n';
    std::cout << avg->GetResult() << '\n';

  #ifdef __CUDNN__
    avg->ForwardPropagateOnGPU();
  #endif  // ifdef __CUDNN__

    std::cout << avg->GetResult()->GetShape() << '\n';
    std::cout << avg->GetResult() << '\n';

    delete input1;
    delete avg;

    return 0;
}
