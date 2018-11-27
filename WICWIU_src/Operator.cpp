
#include "Operator.hpp"

template class Operator<int>;
template class Operator<float>;
template class Operator<double>;

//////////////////////////////////////////////////////////////////////////////// for private method

template<typename DTYPE> int Operator<DTYPE>::Alloc() {
    m_apOutput   = new Container<Operator<DTYPE> *>();
    m_apInput    = new Container<Operator<DTYPE> *>();
    m_aaResult   = new Container<Tensor<DTYPE> *>();
    m_aaGradient = new Container<Tensor<DTYPE> *>();

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::Alloc(int numInput, ...) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
    #endif  // __DEBUG__
    Operator<DTYPE> *temp = NULL;

    va_list ap;
    va_start(ap, numInput);

    int null_count = 0;

    for (int i = 0; i < numInput; i++) {
         temp = va_arg(ap, Operator<DTYPE> *);

         if (!temp) {
           null_count++;
         } else{
           this->AddEdgebetweenOperators(temp);
         }
    }

    va_end(ap);

    if(null_count){
      numInput = numInput - null_count;
      for (int i = 0; i < numInput; i++) {
          delete (*m_apInput)[i];
      }
      delete m_apInput;
      m_apInput = NULL;

      printf("Receive NULL pointer of Operator<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
      return FALSE;
    }

    return TRUE;
}


template<typename DTYPE> void Operator<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__
    int size = 0;

    if (m_aaResult) {
        size = m_aaResult->GetSize();
        Tensor<DTYPE> **ResultContainer = m_aaResult->GetRawData();

        for (int i = 0; i < size; i++) {
            delete ResultContainer[i];
            ResultContainer[i] = NULL;
        }

        delete m_aaResult;
        m_aaResult = NULL;
    }

    if (m_aaGradient) {
        size = m_aaGradient->GetSize();
        Tensor<DTYPE> **GradientContainer = m_aaGradient->GetRawData();

        for (int i = 0; i < size; i++) {
            if ((*m_aaGradient)[i]) {
                delete GradientContainer[i];
                GradientContainer[i] = NULL;
            }
        }

        delete m_aaGradient;
        m_aaGradient = NULL;
    }

    if (m_apOutput) {
        delete m_apOutput;
        m_apOutput = NULL;
    }

    if (m_apInput) {
        delete m_apInput;
        m_apInput = NULL;
    }
}

// Add Graph Edge
template<typename DTYPE> int Operator<DTYPE>::AddInputEdge(Operator<DTYPE> *pInput) {
    try {
        m_apInput->Push(pInput);
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::AddOutputEdge(Operator<DTYPE> *pOutput) {
    try {
        m_apOutput->Push(pOutput);
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    return TRUE;
}

//////////////////////////////////////////////////////////////////////////////// for public method

template<typename DTYPE> Operator<DTYPE>::Operator(std::string pName) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = pName;
    m_Device      = CPU;
    m_Mode        = TRAIN;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    Alloc();
}

template<typename DTYPE> Operator<DTYPE>::Operator(Operator<DTYPE> *pInput, std::string pName) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = pName;
    m_Device      = CPU;
    m_Mode        = TRAIN;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    Alloc();
    AddEdgebetweenOperators(1, pInput);
}

template<typename DTYPE> Operator<DTYPE>::Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, std::string pName) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = pName;
    m_Device      = CPU;
    m_Mode        = TRAIN;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    Alloc();
    AddEdgebetweenOperators(2, pInput0, pInput1);
}

template<typename DTYPE> Operator<DTYPE>::Operator(Operator<DTYPE> *pInput0, Operator<DTYPE> *pInput1, Operator<DTYPE> *pInput2, std::string pName) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = pName;
    m_Device      = CPU;
    m_Mode        = TRAIN;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    Alloc();
    AddEdgebetweenOperators(3, pInput0, pInput1, pInput2);
}

template<typename DTYPE> Operator<DTYPE>::Operator(int numInput, ...) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Operator()" << '\n';
    #endif  // __DEBUG__
    m_apOutput    = NULL;
    m_apInput     = NULL;
    m_aaResult    = NULL;
    m_aaGradient  = NULL;
    m_name        = "";
    m_Device      = CPU;
    m_Mode        = TRAIN;
    m_isParameter = FALSE;
    m_isTrainable = FALSE;
    Alloc();

    va_list ap;
    va_start(ap, numInput);
    AddEdgebetweenOperators(numInput, ap);
    va_end(ap);
}

template<typename DTYPE> Operator<DTYPE>::~Operator() {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::~Operator()" << '\n';
    #endif  // __DEBUG__
    this->Delete();
}

template<typename DTYPE> int Operator<DTYPE>::AddEdgebetweenOperators(Operator<DTYPE> *pInput) {
    this->AddInputEdge(pInput);
    pInput->AddOutputEdge(this);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::AddEdgebetweenOperators(int numInput, va_list ap) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
    #endif  // __DEBUG__
    Operator<DTYPE> *temp = NULL;


    int null_count = 0;

    for (int i = 0; i < numInput; i++) {
         temp = va_arg(ap, Operator<DTYPE> *);

         if (!temp) {
           null_count++;
         } else{
           this->AddEdgebetweenOperators(temp);
         }
    }


    if(null_count){
      numInput = numInput - null_count;
      for (int i = 0; i < numInput; i++) {
          delete (*m_apInput)[i];
      }
      delete m_apInput;
      m_apInput = NULL;

      printf("Receive NULL pointer of Operator<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
      return FALSE;
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::AddEdgebetweenOperators(int numInput, ...) {
    #ifdef __DEBUG__
    std::cout << "Operator<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';
    #endif  // __DEBUG__
    Operator<DTYPE> *temp = NULL;

    va_list ap;
    va_start(ap, numInput);

    int null_count = 0;

    for (int i = 0; i < numInput; i++) {
         temp = va_arg(ap, Operator<DTYPE> *);

         if (!temp) {
           null_count++;
         } else{
           this->AddEdgebetweenOperators(temp);
         }
    }

    va_end(ap);

    if(null_count){
      numInput = numInput - null_count;
      for (int i = 0; i < numInput; i++) {
          delete (*m_apInput)[i];
      }
      delete m_apInput;
      m_apInput = NULL;

      printf("Receive NULL pointer of Operator<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
      return FALSE;
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetResult(Tensor<DTYPE> *pTensor) {
    if (m_aaResult->GetSize()) {
        Tensor<DTYPE> *temp = m_aaResult->Pop();
        delete temp;
        temp = NULL;
    }

    m_aaResult->Push(pTensor);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::AddResult(Tensor<DTYPE> *pTensor) {
    m_aaResult->Push(pTensor);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetGradient(Tensor<DTYPE> *pTensor) {
    if (m_aaGradient->GetSize()) {
        Tensor<DTYPE> *temp = m_aaGradient->Pop();
        delete temp;
        temp = NULL;
    }

    m_aaGradient->Push(pTensor);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::AddGradient(Tensor<DTYPE> *pTensor) {
    m_aaGradient->Push(pTensor);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetDelta(Tensor<DTYPE> *pTensor) {
    if (m_aaGradient->GetSize()) {
        Tensor<DTYPE> *temp = m_aaGradient->Pop();
        delete temp;
        temp = NULL;
    }

    m_aaGradient->Push(pTensor);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::AddDelta(Tensor<DTYPE> *pTensor) {
    m_aaGradient->Push(pTensor);
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetDevice(Device pDevice) {
    m_Device = pDevice;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetDeviceID(unsigned int idOfDevice) {
    m_idOfDevice = idOfDevice;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetIsTensorholder(int pIsParameter) {
    m_isParameter = pIsParameter;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetIsTrainable(int pIsTrainable) {
    m_isTrainable = pIsTrainable;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetModeTrain() {
    m_Mode = TRAIN;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetModeAccumulate() {
    m_Mode = ACCUMULATE;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetModeInference() {
    m_Mode = INFERENCE;
    return TRUE;
}

template<typename DTYPE> Operator<DTYPE> **Operator<DTYPE>::GetOutput() {
    return m_apOutput->GetRawData();
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Operator<DTYPE>::GetOutputContainer() {
    return m_apOutput;
}

template<typename DTYPE> Operator<DTYPE> **Operator<DTYPE>::GetInput() {
    return m_apInput->GetRawData();
}

template<typename DTYPE> Container<Operator<DTYPE> *> *Operator<DTYPE>::GetInputContainer() {
    return m_apInput;
}

template<typename DTYPE> Tensor<DTYPE> *Operator<DTYPE>::GetResult() const {
    return (*m_aaResult)[0];
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Operator<DTYPE>::GetResultContainer() {
    return m_aaResult;
}

template<typename DTYPE> Tensor<DTYPE> *Operator<DTYPE>::GetGradient() const {
    return (*m_aaGradient)[0];
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Operator<DTYPE>::GetGradientContainer() {
    return m_aaGradient;
}

template<typename DTYPE> Tensor<DTYPE> *Operator<DTYPE>::GetDelta() const {
    return (*m_aaGradient)[0];
    // return (*m_aaDelta)[0];
}

template<typename DTYPE> Container<Tensor<DTYPE> *> *Operator<DTYPE>::GetDeltaContainer() {
    return m_aaGradient;
    // return m_aaDelta;
}

template<typename DTYPE> std::string Operator<DTYPE>::GetName() const {
    return m_name;
}

template<typename DTYPE> Device Operator<DTYPE>::GetDevice() {
    return m_Device;
}

template<typename DTYPE> int Operator<DTYPE>::GetDeviceID() {
    return m_idOfDevice;
}

template<typename DTYPE> int Operator<DTYPE>::GetIsTensorholder() {
    return m_isParameter;
}

template<typename DTYPE> int Operator<DTYPE>::GetIsTrainable() {
    return m_isTrainable;
}

template<typename DTYPE> int Operator<DTYPE>::ForwardPropagate(int pTime) {
    #ifdef __DEBUG__

    #endif  // __DEBUG__
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::BackPropagate(int pTime) {
    #ifdef __DEBUG__

    #endif  // __DEBUG__
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::ResetResult() {
    // Tensorholder의 경우는 하면 안된다.
    int size = m_aaResult->GetSize();

    if (m_Device == CPU) {
        for (int i = 0; i < size; i++) {
            (*m_aaResult)[i]->Reset();
        }
    }

#ifdef __CUDNN__
    else if (m_Device == GPU) {
        for (int i = 0; i < size; i++) {
            (*m_aaResult)[i]->Reset(this->GetCudnnHandle());
        }
    }
#endif  // if __CUDNN__

    else return FALSE;

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::ResetGradient() {
    int size = m_aaGradient->GetSize();

    if (m_Device == CPU) {
        for (int i = 0; i < size; i++) {
            (*m_aaGradient)[i]->Reset();
        }
    }

#ifdef __CUDNN__
    else if (m_Device == GPU) {
        for (int i = 0; i < size; i++) {
            (*m_aaGradient)[i]->Reset(this->GetCudnnHandle());
        }
    }
#endif  // if __CUDNN__

    else return FALSE;

    return TRUE;
}

// template<typename DTYPE> void Operator<DTYPE>::PrintInformation() {
//     std::cout << this->GetName() << " : ";
//     std::cout << this->GetResult()->GetShape() << '\n';
// }

template<typename DTYPE> void Operator<DTYPE>::PrintInformation(int level) {
    for(int j = 0; j < level; j++){
        std::cout << "-- ";
    }
    std::cout << this->GetName() << " : ";
    std::cout << this->GetResult()->GetShape() << '\n';
}

template<typename DTYPE> void Operator<DTYPE>::SetDeviceCPU() {
    this->SetDevice(CPU);

    this->SetResultOnCPU();
    this->SetGradientOnCPU();
}

template<typename DTYPE> int Operator<DTYPE>::SetResultOnCPU() {
    // Tensorholder의 경우는 하면 안된다.
    int size = m_aaResult->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaResult)[i]->SetDeviceCPU();
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetGradientOnCPU() {
    int size = m_aaGradient->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaGradient)[i]->SetDeviceCPU();
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::Save(FILE *fileForSave) {
    int size = m_aaResult->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaResult)[i]->Save(fileForSave);
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::Load(FILE *fileForLoad) {
    int size = m_aaResult->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaResult)[i]->Load(fileForLoad);
    }

    return TRUE;
}

#ifdef __CUDNN__

template<typename DTYPE> int Operator<DTYPE>::SetCudnnHandle(cudnnHandle_t& pCudnnHandle) {
    m_pCudnnHandle = pCudnnHandle;
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetResultOnGPU(unsigned int idOfDevice) {
    // Tensorholder의 경우는 하면 안된다.
    int size = m_aaResult->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaResult)[i]->SetDeviceGPU(idOfDevice);
    }

    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::SetGradientOnGPU(unsigned int idOfDevice) {
    int size = m_aaGradient->GetSize();

    for (int i = 0; i < size; i++) {
        (*m_aaGradient)[i]->SetDeviceGPU(idOfDevice);
    }

    return TRUE;
}

template<typename DTYPE> void Operator<DTYPE>::InitializeAttributeForGPU(unsigned int idOfDevice) {}

// template<typename DTYPE> void Operator<DTYPE>::SetDeviceGPU(unsigned int idOfDevice) {
// this->SetDevice(GPU);
// this->SetDeviceID(idOfDevice);
// this->SetResultOnGPU(idOfDevice);
// this->SetGradientOnGPU(idOfDevice);
// }

template<typename DTYPE> void Operator<DTYPE>::SetDeviceGPU(cudnnHandle_t& pCudnnHandle, unsigned int idOfDevice) {
    checkCudaErrors(cudaSetDevice(idOfDevice));
    this->SetCudnnHandle(pCudnnHandle);
    this->SetDevice(GPU);
    this->SetDeviceID(idOfDevice);
    this->SetResultOnGPU(idOfDevice);
    this->SetGradientOnGPU(idOfDevice);
    this->InitializeAttributeForGPU(idOfDevice);
}

template<typename DTYPE> cudnnHandle_t& Operator<DTYPE>::GetCudnnHandle() {
    return m_pCudnnHandle;
}

template<typename DTYPE> int Operator<DTYPE>::ForwardPropagateOnGPU(int pTime) {
    # if __DEBUG__
    std::cout << "Operator<DTYPE>::ForwardPropagateOnGPU(int)" << '\n';
    std::cout << this->GetName() << '\n';
    # endif // __DEBUG__
    return TRUE;
}

template<typename DTYPE> int Operator<DTYPE>::BackPropagateOnGPU(int pTime) {
    # if __DEBUG__
    std::cout << "Operator<DTYPE>::BackPropagateOnGPU(int)" << '\n';
    std::cout << this->GetName() << '\n';
    # endif // __DEBUG__
    return TRUE;
}

#endif  // __CUDNN__


// int main(int argc, char const *argv[]) {
// Operator<int> *temp1 = new Operator<int>("temp1");
// Operator<int> *temp2 = new Operator<int>(temp1, "temp2");
// Operator<int> *temp3 = new Operator<int>(temp1, temp2, "temp3");
//
// std::cout << temp3->GetInput()[0]->GetName() << '\n';
// std::cout << temp3->GetInput()[1]->GetName() << '\n';
// std::cout << temp1->GetOutput()[0]->GetName() << '\n';
//
// delete temp1;
// delete temp2;
// delete temp3;
//
// return 0;
// }
