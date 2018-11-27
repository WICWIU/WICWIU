#include "../WICWIU_src/NeuralNetwork.hpp"


int main(int argc, char const *argv[]) {
    Tensorholder<float> *input0 = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 1, 2, 2, 0.0, 0.1), "x");
    Tensorholder<float> *input1 = new Tensorholder<float>(Tensor<float>::Random_normal(1, 1, 3, 2, 2, 0.0, 0.1), "label");

    std::cout << input0->GetResult()->GetShape() << '\n';
    std::cout << input0->GetResult() << '\n';

    std::cout << input1->GetResult()->GetShape() << '\n';
    std::cout << input1->GetResult() << '\n';

    Operator<float> *concat = new ConcatenateChannelWise<float>(input0, input1);

  #ifdef __CUDNN__
    cudnnHandle_t m_cudnnHandle;
    cudnnCreate(&m_cudnnHandle);
    input0->SetDeviceGPU(m_cudnnHandle, 0);
    input1->SetDeviceGPU(m_cudnnHandle, 0);
    concat->SetDeviceGPU(m_cudnnHandle, 0);
  #endif  // ifdef __CUDNN__

    std::cout << concat->GetResult()->GetShape() << '\n';
    std::cout << concat->GetResult() << '\n';

#ifdef __CUDNN__
    concat->ForwardPropagateOnGPU();
#else // ifdef __CUDNN__
    concat->ForwardPropagate();
#endif  // ifdef __CUDNN__


    std::cout << concat->GetResult()->GetShape() << '\n';
    std::cout << concat->GetResult() << '\n';

    for(int i = 0; i < 2 * 2 * 4; i++){
      (*(concat->GetDelta()))[i] = i;
    }

    std::cout << concat->GetDelta()->GetShape() << '\n';
    std::cout << concat->GetDelta() << '\n';

    std::cout << input0->GetDelta()->GetShape() << '\n';
    std::cout << input0->GetDelta() << '\n';

    std::cout << input1->GetDelta()->GetShape() << '\n';
    std::cout << input1->GetDelta() << '\n';

  #ifdef __CUDNN__
      concat->BackPropagateOnGPU();
  #else // ifdef __CUDNN__
      concat->BackPropagate();
  #endif  // ifdef __CUDNN__

    std::cout << input0->GetDelta()->GetShape() << '\n';
    std::cout << input0->GetDelta() << '\n';

    std::cout << input1->GetDelta()->GetShape() << '\n';
    std::cout << input1->GetDelta() << '\n';

    delete input0;
    delete input1;
    delete concat;

    return 0;
}
