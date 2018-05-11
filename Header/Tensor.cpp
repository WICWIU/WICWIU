#include "Tensor.h"

template class Tensor<int>;
template class Tensor<float>;
template class Tensor<double>;

////////////////////////////////////////////////////////////////////Class Tensor
template<typename DTYPE> Tensor<DTYPE>::Tensor() {
    std::cout << "Tensor::Tensor()" << '\n';
    m_aShape = NULL;
    m_aData  = NULL;
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
    // std::cout << "Tensor::Tensor(int, int, int, int, int)" << '\n';
    m_aShape = NULL;
    m_aData  = NULL;
    Alloc(new Shape(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize));
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(Shape *pShape) {
    std::cout << "Tensor::Tensor(Shape*)" << '\n';
    m_aShape = NULL;
    m_aData  = NULL;
    Alloc(pShape);
}

template<typename DTYPE> Tensor<DTYPE>::Tensor(Tensor *pTensor) {
    std::cout << "Tensor::Tensor(Shape*)" << '\n';
    m_aShape = NULL;
    m_aData  = NULL;
    Alloc(pTensor);
}

template<typename DTYPE> Tensor<DTYPE>::~Tensor() {
    // std::cout << "Tensor::~Tensor()" << '\n';
    Delete();
}

template<typename DTYPE> int Tensor<DTYPE>::Alloc(Shape *pShape) {
    if (pShape == NULL) {
        printf("Receive NULL pointer of Shape class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape = pShape;

        int rank = pShape->GetRank();

        if (rank < 5) {
            delete m_aShape;
            m_aShape = NULL;
            printf("Receive invalid rank value %d in %s (%s %d)\n", rank, __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        } else {
            int capacity = 1;

            for (int i = 0; i < rank; i++) {
                capacity *= (*pShape)[i];
            }
            m_aData = new Data<DTYPE>(capacity);
        }
    }

    return TRUE;
}

template<typename DTYPE> int Tensor<DTYPE>::Alloc(Tensor<DTYPE> *pTensor) {
    if (pTensor == NULL) {
        printf("Receive NULL pointer of Tensor<DTYPE> class in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        m_aShape = new Shape(pTensor->GetShape());
        m_aData  = new Data<DTYPE>(pTensor->GetData());
    }

    return TRUE;
}

template<typename DTYPE> void Tensor<DTYPE>::Delete() {
    if (m_aShape) {
        delete m_aShape;
        m_aShape = NULL;
    }

    if (m_aData) {
        delete m_aData;
        m_aData = NULL;
    }
}

template<typename DTYPE> Shape *Tensor<DTYPE>::GetShape() {
    return m_aShape;
}

template<typename DTYPE> Data<DTYPE> *Tensor<DTYPE>::GetData() {
    return m_aData;
}

template<typename DTYPE> int Tensor<DTYPE>::GetTimeSize() {
    return (*m_aShape)[0];
}

template<typename DTYPE> int Tensor<DTYPE>::GetBatchSize() {
    return (*m_aShape)[1];
}

template<typename DTYPE> int Tensor<DTYPE>::GetChannelSize() {
    return (*m_aShape)[2];
}

template<typename DTYPE> int Tensor<DTYPE>::GetRowSize() {
    return (*m_aShape)[3];
}

template<typename DTYPE> int Tensor<DTYPE>::GetColSize() {
    return (*m_aShape)[4];
}

template<typename DTYPE> int Tensor<DTYPE>::GetCapacity() {
    return m_aData->GetCapacity();
}

template<typename DTYPE> DTYPE& Tensor<DTYPE>::GetRawData() {
    return m_aData->GetRawData();
}

//////////////////////////////////////////////////////////////////

template<typename DTYPE> int Tensor<DTYPE>::Reshape(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
    int cur_capacity = GetCapacity();
    int new_capacity = pTimeSize * pBatchSize * pChannelSize * pRowSize * pColSize;

    if (cur_capacity != new_capacity) {
        printf("Receive invalid shape value in %s (%s %d), cannot Reshape\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    } else {
        (*m_aShape)[0] = pTimeSize;
        (*m_aShape)[1] = pBatchSize;
        (*m_aShape)[2] = pChannelSize;
        (*m_aShape)[3] = pRowSize;
        (*m_aShape)[4] = pColSize;
    }

    return TRUE;
}

template<typename DTYPE> void Tensor<DTYPE>::Reset() {
    int capacity = GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*m_aData)[i] = 0;
    }
}

///////////////////////////////////////////////////////////////////

template<typename DTYPE> DTYPE& Tensor<DTYPE>::operator[](unsigned int index) {
    return (*m_aData)[index];
}

//////////////////////////////////////////////////////////////////static method

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Truncated_normal(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, float mean, float stddev) {
    std::cout << "Tensor<DTYPE>::Truncated_normal()" << '\n';

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> rand(mean, stddev);

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);

    int capacity = temp->GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*temp)[i] = rand(gen);
    }

    return temp;
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Zeros(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize) {
    std::cout << "Tensor<DTYPE>::Zero()" << '\n';

    return new Tensor<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Constants(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize, DTYPE constant) {
    std::cout << "Tensor<DTYPE>::Constant()" << '\n';

    Tensor<DTYPE> *temp = new Tensor<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize);

    int capacity = temp->GetCapacity();

    for (int i = 0; i < capacity; i++) {
        (*temp)[i] = constant;
    }

    return temp;
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Add(Tensor<DTYPE> *pLeftTensor, Tensor<DTYPE> *pRightTensor, Tensor<DTYPE> *pDestTensor) {
    Shape *leftTenShape = pLeftTensor->GetShape();
    int    capacity     = pLeftTensor->GetCapacity();

    int timesize    = (*leftTenShape)[0];
    int batchsize   = (*leftTenShape)[1];
    int channelsize = (*leftTenShape)[2];
    int rowsize     = (*leftTenShape)[3];
    int colsize     = (*leftTenShape)[4];

    if (pDestTensor == NULL) pDestTensor = new Tensor<DTYPE>(timesize,
                                                             batchsize,
                                                             channelsize,
                                                             rowsize,
                                                             colsize);

    for (int i = 0; i < capacity; i++) {
        (*pDestTensor)[i] = (*pLeftTensor)[i] + (*pRightTensor)[i];
    }

    return pDestTensor;
}

template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::BroadcastAdd(Tensor<DTYPE> *pLeftTensor, Tensor<DTYPE> *pRightTensor, Tensor<DTYPE> *pDestTensor, int is_inverse) {
    Shape *leftTenShape  = pLeftTensor->GetShape();
    Shape *rightTenShape = pRightTensor->GetShape();

    int timesize    = (*leftTenShape)[0];
    int batchsize   = (*leftTenShape)[1];
    int channelsize = (*leftTenShape)[2];
    int rowsize     = (*leftTenShape)[3];
    int colsize     = (*leftTenShape)[4];

    if (pDestTensor == NULL) pDestTensor = new Tensor<DTYPE>(timesize,
                                                             batchsize,
                                                             channelsize,
                                                             rowsize,
                                                             colsize);

    int ti = 0;
    int ba = 0;
    int ch = 0;
    int ro = 0;
    int co = 0;

    int zero = 0;

    int *ti_right = &ti;
    int *ba_right = &ba;
    int *ch_right = &ch;
    int *ro_right = &ro;
    int *co_right = &co;

    if ((*rightTenShape)[0] == 1) ti_right = &zero;

    if ((*rightTenShape)[1] == 1) ba_right = &zero;

    if ((*rightTenShape)[2] == 1) ch_right = &zero;

    if ((*rightTenShape)[3] == 1) ro_right = &zero;

    if ((*rightTenShape)[4] == 1) co_right = &zero;

    for (ti = 0; ti < timesize; ti++) {
        for (ba = 0; ba < batchsize; ba++) {
            for (ch = 0; ch < channelsize; ch++) {
                for (ro = 0; ro < rowsize; ro++) {
                    for (co = 0; co < colsize; co++) {
                        (*pDestTensor)[Index5D(leftTenShape, ti, ba, ch, ro, co)]
                            = (*pLeftTensor)[Index5D(leftTenShape, ti, ba, ch, ro, co)]
                              + (*pRightTensor)[Index5D(rightTenShape, *ti_right, *ba_right, *ch_right, *ro_right, *co_right)];
                    }
                }
            }
        }
    }

    return pDestTensor;
}


template<typename DTYPE> Tensor<DTYPE> *Tensor<DTYPE>::Multiply(Tensor<DTYPE> *pLeftTensor, float pMultiplier, Tensor<DTYPE> *pDestTensor) {
    Shape *leftTenShape = pLeftTensor->GetShape();
    int    capacity     = pLeftTensor->GetCapacity();

    int timesize    = (*leftTenShape)[0];
    int batchsize   = (*leftTenShape)[1];
    int channelsize = (*leftTenShape)[2];
    int rowsize     = (*leftTenShape)[3];
    int colsize     = (*leftTenShape)[4];

    if (pDestTensor == NULL) pDestTensor = new Tensor<DTYPE>(timesize,
                                                             batchsize,
                                                             channelsize,
                                                             rowsize,
                                                             colsize);

    for (int i = 0; i < capacity; i++) {
        (*pDestTensor)[i] = (*pLeftTensor)[i] * pMultiplier;
    }

    return pDestTensor;
}
// example code
// int main(int argc, char const *argv[]) {
// Tensor<float> *left  = Tensor<float>::Constants(1, 2, 3, 3, 3, 2);
// Tensor<float> *right = Tensor<float>::Truncated_normal(1, 1, 3, 1, 1, 0.0, 0.1);
// Tensor<float> *dst   = Tensor<float>::Zeros(1, 2, 3, 3, 3);
//
// std::cout << left << '\n';
// std::cout << right << '\n';
// std::cout << dst << '\n';
//
// Tensor<float>::BroadcastAdd(left, right);
//
// std::cout << dst << '\n';
//
// return 0;
// }
