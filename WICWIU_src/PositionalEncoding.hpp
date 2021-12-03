#ifndef POSITIONALENCODING_H_
#define POSITIONALENCODING_H_ value

#include "Tensor.hpp"
template <typename DTYPE> class PositionalEncoding {
private:
    Tensor<DTYPE> *m_aPositionalEncoding;

public:
    PositionalEncoding(int batchSize, int vocabLength, int embeddingSize);
    virtual ~PositionalEncoding();

    int  Alloc(int batchSize, int vocabLength, int embeddingSize);
    void Delete();

    Tensor<DTYPE> *GetPositionalEncoding();
};

template <typename DTYPE> PositionalEncoding<DTYPE>::PositionalEncoding(int batchSize, int vocabLength, int embeddingSize) {
#ifdef __DEBUG__
    std::cout << "PositionalEncoding::PositionalEncoding(int batchSize, int vocabLength, int embeddingSize)" << '\n';
#endif // __DEBUG__
    this->Alloc(batchSize, vocabLength, embeddingSize);
}

template <typename DTYPE> PositionalEncoding<DTYPE>::~PositionalEncoding() {
#ifdef __DEBUG__
    std::cout << "PositionalEncoding::~PositionalEncoding()" << '\n';
#endif // __DEBUG__
    Delete();
}

template <typename DTYPE> int PositionalEncoding<DTYPE>::Alloc(int batchSize, int vocabLength, int embeddingSize) {
#ifdef __DEBUG__
    std::cout << "PositionalEncoding::Alloc(batchSize, vocabLength, embeddingSize)" << '\n';
#endif // __DEBUG__
    m_aPositionalEncoding = Tensor<DTYPE>::Zeros(1, batchSize, vocabLength, 1, embeddingSize);

    return TRUE;
}

template <typename DTYPE> void PositionalEncoding<DTYPE>::Delete() {
    if (m_aPositionalEncoding)
        delete m_aPositionalEncoding;
}

template <typename DTYPE> Tensor<DTYPE> *PositionalEncoding<DTYPE>::GetPositionalEncoding() {
    return m_aPositionalEncoding;
}

template <typename DTYPE> class TransformerPositionalEncoding : public PositionalEncoding<DTYPE> {
public:
    TransformerPositionalEncoding(int batchSize, int vocabLength, int embeddingSize);
    virtual ~TransformerPositionalEncoding();

    int  Alloc();
    void Delete();
};

template <typename DTYPE> TransformerPositionalEncoding<DTYPE>::TransformerPositionalEncoding(int batchSize, int vocabLength, int embeddingSize) : PositionalEncoding<DTYPE>(batchSize, vocabLength, embeddingSize) {
#ifdef __DEBUG__
    std::cout << "TransformerPositionalEncoding::TransformerPositionalEncoding(int batchSize, int vocabLength, int embeddingSize)" << '\n';
#endif // __DEBUG__
    this->Alloc();
}

template <typename DTYPE> TransformerPositionalEncoding<DTYPE>::~TransformerPositionalEncoding() {
#ifdef __DEBUG__
    std::cout << "TransformerPositionalEncoding::~TransformerPositionalEncoding()" << '\n';
#endif // __DEBUG__
    Delete();
}

template <typename DTYPE> int TransformerPositionalEncoding<DTYPE>::Alloc() {
    Tensor<DTYPE> *pPositionalEncoding = this->GetPositionalEncoding();
    int            pTimeSize           = pPositionalEncoding->GetTimeSize();
    int            pBatchSize          = pPositionalEncoding->GetBatchSize();
    int            pChannelSize        = pPositionalEncoding->GetChannelSize();
    int            pRowSize            = pPositionalEncoding->GetRowSize();
    int            pColumnSize         = pPositionalEncoding->GetColSize();

    Shape *peShape = pPositionalEncoding->GetShape();

    for (int ti = 0; ti < pTimeSize; ti++) {
        for (int ba = 0; ba < pBatchSize; ba++) {
            for (int ch = 0; ch < pChannelSize; ch++) {
                for (int ro = 0; ro < pRowSize; ro++) {
                    for (int co = 0; co < pColumnSize; co++) {
                        int   i = 0, pos = 0;
                        float div_term = 0, pe = 0;
                        i   = co / 2;
                        pos = ch;
                        div_term = pow(10000, 2 * i / (double)pColumnSize);
                        if (co % 2 == 0)
                            pe = sin(pos / div_term);
                        else
                            pe = cos(pos / div_term);
                        (*pPositionalEncoding)[Index5D(peShape, ti, ba, ch, ro, co)] = pe;
                    }
                }
            }
        }
    }

    return TRUE;
}

template <typename DTYPE> void TransformerPositionalEncoding<DTYPE>::Delete() {
#ifdef __DEBUG__
    std::cout << "Transformer Postional Encoding Delete()" << '\n';
#endif //__DEBUG__
}

#endif // Embedding_H_
