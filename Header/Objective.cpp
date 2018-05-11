#include "Objective.h"

template class Objective<int>;
template class Objective<float>;
template class Objective<double>;

template<typename DTYPE> Objective<DTYPE>::Objective(std::string pName) {
    std::cout << "Objective<DTYPE>::Objective()" << '\n';
    m_aResult        = NULL;
    m_aGradient      = NULL;
    m_pInputOperator = NULL;
    m_pInputTensor   = NULL;
    m_pLabel         = NULL;
    m_name           = pName;
}

template<typename DTYPE> Objective<DTYPE>::Objective(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) {
    std::cout << "Objective<DTYPE>::Objective()" << '\n';
    m_aResult        = NULL;
    m_aGradient      = NULL;
    m_pInputOperator = NULL;
    m_pInputTensor   = NULL;
    m_pLabel         = NULL;
    m_name           = pName;
    Alloc(pOperator, pLabel);
}

template<typename DTYPE> Objective<DTYPE>::~Objective() {
    std::cout << "Objective<DTYPE>::~Objective()" << '\n';
    this->Delete();
}

template<typename DTYPE> int Objective<DTYPE>::Alloc(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel) {
    std::cout << "Objective<DTYPE>::Alloc(Tensor<DTYPE> *)" << '\n';

    m_pInputOperator = pOperator;
    m_pInputTensor   = m_pInputOperator->GetResult();

    m_pLabel = pLabel;
    return TRUE;
}

template<typename DTYPE> void Objective<DTYPE>::Delete() {
    if (m_aResult) {
        delete m_aResult;
        m_aResult = NULL;
    }

    if (m_aGradient) {
        delete m_aGradient;
        m_aGradient = NULL;
    }
}

template<typename DTYPE> void Objective<DTYPE>::SetResult(Tensor<DTYPE> *pTensor) {
    m_aResult = pTensor;
}

template<typename DTYPE> void Objective<DTYPE>::SetGradient(Tensor<DTYPE> *pTensor) {
    m_aGradient = pTensor;
}

template<typename DTYPE> Tensor<DTYPE> *Objective<DTYPE>::GetResult() const {
    return m_aResult;
}

template<typename DTYPE> Tensor<DTYPE> *Objective<DTYPE>::GetGradient() const {
    return m_aGradient;
}

template<typename DTYPE> Operator<DTYPE> *Objective<DTYPE>::GetOperator() const {
    return m_pInputOperator;
}

template<typename DTYPE> Tensor<DTYPE> *Objective<DTYPE>::GetTensor() const {
    return m_pInputTensor;
}

template<typename DTYPE> Operator<DTYPE> *Objective<DTYPE>::GetLabel() const {
    return m_pLabel;
}

template<typename DTYPE> std::string Objective<DTYPE>::GetName() const {
    return m_name;
}

template<typename DTYPE> Tensor<DTYPE> *Objective<DTYPE>::ForwardPropagate() {
    std::cout << this->GetName() << '\n';
    return NULL;
}

template<typename DTYPE> Tensor<DTYPE> *Objective<DTYPE>::BackPropagate() {
    std::cout << this->GetName() << '\n';
    return NULL;
}

template<typename DTYPE> DTYPE& Objective<DTYPE>::operator[](unsigned int index) {
    return (*m_aResult)[index];
}

template<typename DTYPE> int Objective<DTYPE>::ResetResult() {
    m_aResult->Reset();

    return TRUE;
}

template<typename DTYPE> int Objective<DTYPE>::ResetGradient() {
    m_aGradient->Reset();

    return TRUE;
}

// int main(int argc, char const *argv[]) {
// Objective<int> *temp1 = new Objective<int>("temp1");
// Objective<int> *temp2 = new Objective<int>(temp1, "temp2");
// Objective<int> *temp3 = new Objective<int>(temp1, temp2, "temp3");
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
