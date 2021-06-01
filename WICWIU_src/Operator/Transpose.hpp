#ifndef TRANSPOSE_HPP_
#define TRANSPOSE_HPP_    value

#include "../Operator.hpp"

/*!
@class Transpose Operator의 input result를 Transpose를 시켜 Operator의 result에 저장하는 클래스
@details Operator의 input result에 Tensor를 원하는 2개의 Dim을 선택하여 Transpose 기능을 수행한다
@details 단, LongArray 구조적 특성상 Time축은 지원하지 않고 나머지 Batch, Channel, Row, Column 축 안에서의 Transpose만 지원한다.
*/

template<typename DTYPE> class Transpose : public Operator<DTYPE>{
private:
    int m_pDim0;
    int m_pDim1;
    int* m_aNewDim;
public:
    Transpose(Operator<DTYPE> *pInput, int dim0, int dim1, std::string pName, int pLoadflag = TRUE) : Operator<DTYPE>(pInput, pName, pLoadflag) {
        #ifdef __DEBUG__
        std::cout << "Transpose::Transpose(Operator *)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput, dim0, dim1);
    }
    ~Transpose() {
        #ifdef __DEBUG__
        std::cout << "Transpose::~Transpose()" << '\n';
        #endif  // __DEBUG__

        Delete();
    }
    int Alloc(Operator<DTYPE> *pInput, int dim0, int dim1) {
        #ifdef __DEBUG__
        std::cout << "Transpose::Alloc(Operator *, Operator *)" << '\n';
        #endif  // __DEBUG__
        if(dim0<0) dim0 += 5;
        if(dim1<0) dim1 += 5;

        if(dim0<0 || dim0>4 || dim1<0 || dim1>4){
            std::cout << "Out of Range!" << '\n';
            return FALSE;
        }

        if(dim0>=dim1){
            m_pDim0 = dim0;
            m_pDim1 = dim1;
        }
        else{
            m_pDim0 = dim1;
            m_pDim1 = dim0;
        }

        Shape *pInputShape = pInput->GetResult()->GetShape();
        int pDim0Size = (*pInputShape)[m_pDim0];
        int pDim1Size = (*pInputShape)[m_pDim1];
        try {
            m_aNewDim = new int[5];
        } catch (...) {
            printf("Failed to allcate memory in %s (%s %d)\n", __FUNCTION__, __FILE__, __LINE__);
            return FALSE;
        }
        for(int i=0; i<5; i++){
            m_aNewDim[i] = pInputShape->GetDim(i);
        }
        m_aNewDim[m_pDim1] = pDim0Size;
        m_aNewDim[m_pDim0] = pDim1Size;

        Tensor<DTYPE> *result = new Tensor<DTYPE>(m_aNewDim[0],m_aNewDim[1],m_aNewDim[2],m_aNewDim[3],m_aNewDim[4]);
        this->SetResult(result);
        Tensor<DTYPE> *delta = new Tensor<DTYPE>(m_aNewDim[0],m_aNewDim[1],m_aNewDim[2],m_aNewDim[3],m_aNewDim[4]);
        this->SetDelta(delta);

        return TRUE;
    }

    void Delete() {
        delete[] m_aNewDim;
    }

    int  ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *pInputTensor = this->GetInput()[0]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        Shape *pInputShape = pInputTensor->GetShape();
        Shape *pResultShape = this->GetResult()->GetShape();

        int* pStride;
        pStride = new int[5];

        for(int i=4; i>=0; i--){
            if(i==4) pStride[i] = 1;
            else pStride[i] = pStride[i+1]*(pInputShape->GetDim(i+1));
        }

        int temp = pStride[m_pDim0];
        pStride[m_pDim0] = pStride[m_pDim1];
        pStride[m_pDim1] = temp;

        int t_pTimeSize    = pResultShape->GetDim(0);
        int t_pBatchSize   = pResultShape->GetDim(1);
        int t_pChannelSize = pResultShape->GetDim(2);
        int t_pRowSize     = pResultShape->GetDim(3);
        int t_pColumnSize  = pResultShape->GetDim(4);

        for(int ti=0; ti<t_pTimeSize; ti++){
            for(int ba=0; ba<t_pBatchSize; ba++){
                for(int ch=0; ch<t_pChannelSize; ch++){
                    for(int ro=0; ro<t_pRowSize; ro++){
                        for(int co=0; co<t_pColumnSize; co++){
                            int index = ti*pStride[0]+ba*pStride[1]+ch*pStride[2]+ro*pStride[3]+co*pStride[4];
                            (*result)[Index5D(pResultShape, ti, ba, ch, ro, co)] = (*pInputTensor)[index];
                        }
                    }
                }
            }
        }

        delete[] pStride;

        return TRUE;
    }

    int BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *input_delta     = this->GetInput()[0]->GetDelta();
        Tensor<DTYPE> *this_delta      = this->GetDelta();

        Shape *pDeltaShape = this_delta->GetShape();
        Shape *pInputDeltaShape = input_delta->GetShape();

        int* pStride;
        pStride = new int[5];

        for(int i=4; i>=0; i--){
            if(i==4) pStride[i] = 1;
            else pStride[i] = pStride[i+1]*(pDeltaShape->GetDim(i+1));
            //std::cout << pStride[i] << " ";
        }
        int temp = pStride[m_pDim0];
        pStride[m_pDim0] = pStride[m_pDim1];
        pStride[m_pDim1] = temp;

        int pTimeSize    = pInputDeltaShape->GetDim(0);
        int pBatchSize   = pInputDeltaShape->GetDim(1);
        int pChannelSize = pInputDeltaShape->GetDim(2);
        int pRowSize     = pInputDeltaShape->GetDim(3);
        int pColumnSize  = pInputDeltaShape->GetDim(4);

        for(int ti=0; ti<pTimeSize; ti++){
            for(int ba=0; ba<pBatchSize; ba++){
                for(int ch=0; ch<pChannelSize; ch++){
                    for(int ro=0; ro<pRowSize; ro++){
                        for(int co=0; co<pColumnSize; co++){
                            int index = ti*pStride[0]+ba*pStride[1]+ch*pStride[2]+ro*pStride[3]+co*pStride[4];
                            (*input_delta)[Index5D(pInputDeltaShape, ti, ba, ch, ro, co)] = (*this_delta)[index];
                        }
                    }
                }
            }
        }

        delete[] pStride;

        return TRUE;
    }
};

#endif  // Transpose_H_
