#include "Shape.h"

Shape::Shape() {
    m_Rank = 0;
    m_aDim = NULL;
#if __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__
}

Shape::Shape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4) {
    #if __DEBUG__
    std::cout << "Shape::Shape(int, int, int, int, int)" << '\n';
    #endif  // __DEBUG__
    m_Rank = 0;
    m_aDim = NULL;
#if __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__
    Alloc(5, pSize0, pSize1, pSize2, pSize3, pSize4);
}

Shape::Shape(Shape *pShape) {
    #if __DEBUG__
    std::cout << "Shape::Shape(Shape *)" << '\n';
    #endif  // __DEBUG__
    m_Rank = 0;
    m_aDim = NULL;
#if __CUDNN__
    m_desc = NULL;
#endif  // if __CUDNN__
    Alloc(pShape);
}

Shape::~Shape() {
    // std::cout << "Shape::~Shape()" << '\n';
    Delete();
}

int Shape::Alloc() {
    m_Rank = 0;
    m_aDim = NULL;
#if __CUDNN__
    checkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
#endif  // if __CUDNN__
    return TRUE;
}

int Shape::Alloc(int pRank, ...) {
    try {
        if (pRank > 0) m_Rank = pRank;
        else throw pRank;
    } catch (int e) {
        printf("Receive invalid rank value %d in %s (%s %d)\n", e, __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    try {
        m_aDim = new int[m_Rank];
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    va_list ap;
    va_start(ap, pRank);

    // need to check compare between pRank value and number of another parameter
    for (int i = 0; i < pRank; i++) {
        // need to check whether int or not
        m_aDim[i] = va_arg(ap, int);
    }
    va_end(ap);

#if __CUDNN__

    if (m_Rank == 5) {
        checkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
        checkCUDNN(cudnnSetTensor4dDescriptor(m_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              m_aDim[1], m_aDim[2], m_aDim[3], m_aDim[4]));
    }

#endif  // if __CUDNN__

    return TRUE;
}

int Shape::Alloc(Shape *pShape) {
    try {
        m_Rank = pShape->GetRank();
        m_aDim = new int[m_Rank];

        for (int i = 0; i < m_Rank; i++) {
            m_aDim[i] = (*pShape)[i];
        }

#if __CUDNN__

        if (m_Rank == 5) {
            checkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
            checkCUDNN(cudnnSetTensor4dDescriptor(m_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                  m_aDim[1], m_aDim[2], m_aDim[3], m_aDim[4]));
        }

#endif  // if __CUDNN__
    } catch (...) {
        printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        return FALSE;
    }

    return TRUE;
}

void Shape::Delete() {
    if (m_aDim) {
        delete[] m_aDim;
        m_aDim = NULL;
    }

#if __CUDNN__

    if (m_desc) checkCUDNN(cudnnDestroyTensorDescriptor(m_desc));
    m_desc = NULL;
#endif  // if __CUDNN__
}

void Shape::SetRank(int pRank) {
    m_Rank = pRank;
}

int Shape::GetRank() {
    return m_Rank;
}

int Shape::ReShape(int pSize0, int pSize1, int pSize2, int pSize3, int pSize4) {
    m_aDim[0] = pSize0;
    m_aDim[1] = pSize1;
    m_aDim[2] = pSize2;
    m_aDim[3] = pSize3;
    m_aDim[4] = pSize4;

#if __CUDNN__

    if (m_desc) checkCUDNN(cudnnDestroyTensorDescriptor(m_desc));
    m_desc = NULL;
    checkCUDNN(cudnnCreateTensorDescriptor(&m_desc));
    checkCUDNN(cudnnSetTensor4dDescriptor(m_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          pSize1, pSize2, pSize3, pSize4));
#endif  // if __CUDNN__

    return TRUE;
}

#if __CUDNN__
cudnnTensorDescriptor_t& Shape::GetDescriptor() {
    return m_desc;
}

#endif  // __CUDNN__

int& Shape::operator[](int pRanknum) {
    try {
        if (pRanknum >= 0) return m_aDim[pRanknum];
        else throw;
    }
    catch (...) {
        printf("Receive invalid pRanknum value in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
        exit(0);
        // return FALSE;
    }
}

/////////////////////////////////////////////////// for Print Shape Information
std::ostream& operator<<(std::ostream& pOS, Shape *pShape) {
    int rank = pShape->GetRank();

    pOS << "Rank is " << rank << ", Dimension is [";

    for (int i = 0; i < rank; i++) pOS << (*pShape)[i] << ", ";
    pOS << "]";
    return pOS;
}

//// example code
// int main(int argc, char const *argv[]) {
// Shape *temp = new Shape(1, 1, 1, 4, 2);
//
// std::cout << *temp << '\n';
//
// Shape *copy = new Shape(temp);
//
// std::cout << *copy << '\n';
//
// delete temp;
// delete copy;
//
// return 0;
// }
