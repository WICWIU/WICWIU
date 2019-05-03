#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "../../WICWIU_src/Tensor.hpp"
#include "../../WICWIU_src/DataLoader.hpp"

#define DIMOFMNISTIMAGE     784
#define DIMOFMNISTLABEL     10

#define NUMOFTESTDATA       10000
#define NUMOFTRAINDATA      60000

using namespace std;

enum OPTION {
    TESTING,
    TRAINING
};

int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;

    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;

    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

template<typename DTYPE> void IMAGE_Reader(string pImagePath, DTYPE **pImage) {
    ifstream fin;
    fin.open(pImagePath, ios_base::binary);

    if (fin.is_open()) {
        int magicNumber = 0;
        int numOfImage  = 0;
        int n_rows      = 0;
        int n_cols      = 0;

        fin.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = ReverseInt(magicNumber);

        fin.read((char *)&numOfImage,  sizeof(numOfImage));
        numOfImage = ReverseInt(numOfImage);

        fin.read((char *)&n_rows,      sizeof(n_rows));
        n_rows = ReverseInt(n_rows);

        fin.read((char *)&n_cols,      sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        int dimOfImage = n_rows * n_cols;

        for (int i = 0; i < numOfImage; ++i) {
            pImage[i] = new DTYPE[dimOfImage];

            for (int d = 0; d < dimOfImage; ++d) {
                unsigned char data = 0;
                fin.read((char *)&data, sizeof(data));
                pImage[i][d] = (DTYPE)data / 255.0;
            }
        }
    }
    fin.close();
}

template<typename DTYPE>
void LABEL_Reader(string pLabelPath, DTYPE **pLabel) {
    ifstream fin;
    fin.open(pLabelPath, ios_base::binary);

    if (fin.is_open()) {
        int magicNumber = 0;
        int numOfLabels = 0;

        fin.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = ReverseInt(magicNumber);

        fin.read((char *)&numOfLabels, sizeof(numOfLabels));
        numOfLabels = ReverseInt(numOfLabels);

        for (int i = 0; i < numOfLabels; ++i) {
            pLabel[i] = new DTYPE[1];
            unsigned char data = 0;
            fin.read((char *)&data, sizeof(data));
            pLabel[i][0] = (DTYPE)data;
        }
    }
    fin.close();
}

template<typename DTYPE>
class MNISTDataSet : public Dataset<DTYPE>{
private:
    DTYPE **m_aaImage;
    DTYPE **m_aaLabel;
    int m_numOfImg;

    OPTION m_option;

public:
    MNISTDataSet(string pImagePath, string pLabelPath, OPTION pOPTION) {
        m_aaImage = NULL;
        m_aaLabel = NULL;

        m_option = pOPTION;

        Alloc(pImagePath, pLabelPath);
    }

    virtual ~MNISTDataSet() {
        Delete();
    }

    virtual void                          Alloc(string pImagePath, string pLabelPath);

    virtual void                          Delete();

    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    virtual int                           GetLength();

};

template<typename DTYPE> void MNISTDataSet<DTYPE>::Alloc(string pImagePath, string pLabelPath) {
    if (m_option == TRAINING) {
        m_numOfImg = NUMOFTRAINDATA;
        m_aaImage = new DTYPE *[m_numOfImg];
        IMAGE_Reader(pImagePath, m_aaImage);
        m_aaLabel = new DTYPE *[m_numOfImg];
        LABEL_Reader(pLabelPath, m_aaLabel);
    } else if (m_option == TESTING) {
        m_numOfImg = NUMOFTESTDATA;
        m_aaImage = new DTYPE *[m_numOfImg];
        IMAGE_Reader(pImagePath, m_aaImage);
        m_aaLabel = new DTYPE *[m_numOfImg];
        LABEL_Reader(pLabelPath, m_aaLabel);
    } else {
        printf("invalid option\n");
        exit(-1);
    }
}

template<typename DTYPE> void MNISTDataSet<DTYPE>::Delete() {
    if (m_aaImage) {
        for (int i = 0; i < m_numOfImg; i++) {
            if (m_aaImage[i]) {
                delete[] m_aaImage[i];
                m_aaImage[i] = NULL;
            }
        }
        delete m_aaImage;
        m_aaImage = NULL;
    }

    if (m_aaLabel) {
        for (int i = 0; i < m_numOfImg; i++) {
            if (m_aaLabel[i]) {
                delete[] m_aaLabel[i];
                m_aaLabel[i] = NULL;
            }
        }
        delete m_aaLabel;
        m_aaLabel = NULL;
    }
}

template<typename DTYPE> std::vector<Tensor<DTYPE> *> *MNISTDataSet<DTYPE>::GetData(int idx) {
    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

    Tensor<DTYPE> *image = Tensor<DTYPE>::Zeros(1, 1, 1, 1, DIMOFMNISTIMAGE);
    Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(1, 1, 1, 1, DIMOFMNISTLABEL);

    for (int i = 0; i < DIMOFMNISTIMAGE; i++) {
        (*image)[i] = m_aaImage[idx][i];
    }

    (*label)[(int)m_aaLabel[idx][0]] = 1.f;

    result->push_back(image);
    result->push_back(label);

    return result;
}

template<typename DTYPE> int MNISTDataSet<DTYPE>::GetLength() {
    if (m_option == TRAINING) {
        return NUMOFTRAINDATA;
    } else if (m_option == TESTING) {
        return NUMOFTESTDATA;
    }
    return 0;
}
