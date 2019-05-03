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

#define TEST_IMAGE_FILE     "data/t10k-images-idx3-ubyte"
#define TEST_LABEL_FILE     "data/t10k-labels-idx1-ubyte"
#define TRAIN_IMAGE_FILE    "data/train-images-idx3-ubyte"
#define TRAIN_LABEL_FILE    "data/train-labels-idx1-ubyte"

using namespace std;

enum OPTION {
    TESTING,
    TESTIMAGE,
    TESTLABEL,
    TRAINING,
    TRAINIMAGE,
    TRAINLABEL,
    DEFAULT
};

int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;

    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;

    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

template<typename DTYPE> void IMAGE_Reader(string DATAPATH, DTYPE **pImage) {
    ifstream fin;

    if (DATAPATH == TEST_IMAGE_FILE) {
        fin.open(TEST_IMAGE_FILE, ios_base::binary);
    } else if (DATAPATH == TRAIN_IMAGE_FILE) {
        fin.open(TRAIN_IMAGE_FILE, ios_base::binary);
    }

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
void LABEL_Reader(string DATAPATH, DTYPE **pLabel) {
    ifstream fin;

    if (DATAPATH == TEST_LABEL_FILE) {
        fin.open(TEST_LABEL_FILE, ios_base::binary);
    } else if (DATAPATH == TRAIN_LABEL_FILE) {
        fin.open(TRAIN_LABEL_FILE, ios_base::binary);
    }

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
DTYPE** ReShapeData(OPTION pOption) {
    if (pOption == TESTIMAGE) {
        DTYPE **TestImage = new DTYPE *[NUMOFTESTDATA];
        IMAGE_Reader(TEST_IMAGE_FILE, TestImage);

        return TestImage;
    } else if (pOption == TESTLABEL) {
        DTYPE **TestLabel = new DTYPE *[NUMOFTESTDATA];
        LABEL_Reader(TEST_LABEL_FILE, TestLabel);

        return TestLabel;
    } else if (pOption == TRAINIMAGE) {
        DTYPE **TrainImage = new DTYPE *[NUMOFTRAINDATA];
        IMAGE_Reader(TRAIN_IMAGE_FILE, TrainImage);

        return TrainImage;
    } else if (pOption == TRAINLABEL) {
        DTYPE **TrainLabel = new DTYPE *[NUMOFTRAINDATA];
        LABEL_Reader(TRAIN_LABEL_FILE, TrainLabel);

        return TrainLabel;
    } else return NULL;
}

template<typename DTYPE>
class MNISTDataSet : public Dataset<DTYPE>{
private:
    DTYPE **m_aaImage;
    DTYPE **m_aaLabel;

    OPTION m_option;

public:
    MNISTDataSet(OPTION pOPTION) {
        m_aaImage = NULL;
        m_aaLabel = NULL;

        m_option = pOPTION;

        Alloc();
    }

    virtual ~MNISTDataSet() {
        Delete();
    }

    virtual void                          Alloc();

    virtual void                          Delete();

    virtual std::vector<Tensor<DTYPE> *>* GetData(int idx);

    virtual int                           GetLength();

    void                                  SetImage(DTYPE **pImage) {
        m_aaImage = pImage;
    }

    void SetLabel(DTYPE **pLabel) {
        m_aaLabel = pLabel;
    }
};

template<typename DTYPE> void MNISTDataSet<DTYPE>::Alloc() {
    if (m_option == TRAINING) {
        this->SetImage(ReShapeData<DTYPE>(TRAINIMAGE));
        this->SetLabel(ReShapeData<DTYPE>(TRAINLABEL));
    } else if (m_option == TESTING) {
        this->SetImage(ReShapeData<DTYPE>(TESTIMAGE));
        this->SetLabel(ReShapeData<DTYPE>(TESTLABEL));
    } else {
        printf("invalid option\n");
        exit(-1);
    }
}

template<typename DTYPE> void MNISTDataSet<DTYPE>::Delete() {
    int numOfImg = 0;

    if (m_option == TRAINING) numOfImg = NUMOFTRAINDATA;
    else numOfImg = NUMOFTESTDATA;

    if (m_aaImage) {
        for (int i = 0; i < numOfImg; i++) {
            if (m_aaImage[i]) {
                delete[] m_aaImage[i];
                m_aaImage[i] = NULL;
            }
        }
        delete m_aaImage;
        m_aaImage = NULL;
    }

    if (m_aaLabel) {
        for (int i = 0; i < numOfImg; i++) {
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
