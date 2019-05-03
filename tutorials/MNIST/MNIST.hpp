#include "../../WICWIU_src/DataLoader.hpp"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>

#include <string.h>

#define DIMOFMNISTIMAGE     784
#define DIMOFMNISTLABEL     10

#define NUMOFTESTDATA       10000
#define NUMOFTRAINDATA      60000

#define TEST_IMAGE_FILE     "data/t10k-images-idx3-ubyte"
#define TEST_LABEL_FILE     "data/t10k-labels-idx1-ubyte"
#define TRAIN_IMAGE_FILE    "data/train-images-idx3-ubyte"
#define TRAIN_LABEL_FILE    "data/train-labels-idx1-ubyte"

enum OPTIONS {
    Train,
    Test
};

template<typename DTYPE> class MNIST : public Dataset<DTYPE>{
private:
    /* data */
    char m_imagePath[256];
    char m_labelPath[256];
    OPTIONS m_option;

public:
    MNIST(char* piamgePath, char* plabelPath, OPTIONS pOPTION = Train);
    ~MNIST();
    void Alloc();
    void Dealloc();
    std::vector<Tensor<DTYPE> *>* GetData(int idx);
    int GetLength(); // all of data Length
    int ReverseInt(DTYPE i);
};

template<typename DTYPE> MNIST<DTYPE>::MNIST(char* piamgePath, char* plabelPath, OPTIONS pOPTION = Train){
    strcpy(m_imagePath, piamgePath); 
    strcpy(m_labelPath, plabelPath); 
    m_option = pOPTION;

    Alloc();
}
template<typename DTYPE> MNIST<DTYPE>::~MNIST(){

    Dealloc();
}

template<typename DTYPE> void MNIST<DTYPE>::Alloc(){

}

template<typename DTYPE> std::vector<Tensor<DTYPE> *>* MNIST<DTYPE>::GetData(int idx){
    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(1, NULL);
    Tensor<DTYPE> **temp = new Tensor<DTYPE> *[2]; 
    int capacity = DIMOFMNISTIMAGE;
    int numOfData = 0;

    std::ifstream fin_data, fin_label;


    Tensor<DTYPE> *data = Tensor<DTYPE>::Zeros(1, 1, 1, 1, capacity);

    if(m_option == Train){
        numOfData = NUMOFTRAINDATA;
    } else if(m_option == Test){
        numOfData = NUMOFTESTDATA;
    }

    fin_data.open(m_imagePath, std::ios_base::binary);
    fin_label.open(m_labelPath, std::ios_base::binary);
    
    if (fin_data.is_open() && fin_label.is_open()){
        //Load Image
        int magicNumber = 0;
        int numOfImage  = 0;
        int n_rows      = 0;
        int n_cols      = 0;

        fin_data.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = ReverseInt(magicNumber);

        fin_data.read((char *)&numOfImage,  sizeof(numOfImage));
        numOfImage = ReverseInt(numOfImage);

        fin_data.read((char *)&n_rows,      sizeof(n_rows));
        n_rows = ReverseInt(n_rows);

        fin_data.read((char *)&n_cols,      sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        int dimOfImage = n_rows * n_cols;
        Tensor<DTYPE> *temp_image = new Tensor<DTYPE>(1, 1, 1, 1, dimOfImage);

        fin_data.seekg(sizeof(unsigned char) * dimOfImage * idx, std::ios::cur);
        
        for (int d = 0; d < dimOfImage; ++d) {
            unsigned char data = 0;
            fin_data.read((char *)&data, sizeof(data));
            (*temp_image)[d] =  (DTYPE)data / 255.0;
            // (*result)[0][d] = (DTYPE)data / 255.0;
        }   
        result->push_back(temp_image);

        //load Label
        magicNumber = 0;
        int numOfLabels = 0;
        Tensor<DTYPE> *temp_label = Tensor<DTYPE>::Zeros(1, 1, 1, 1, DIMOFMNISTLABEL);

        fin_label.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = ReverseInt(magicNumber);

        fin_label.read((char *)&numOfLabels, sizeof(numOfLabels));
        numOfLabels = ReverseInt(numOfLabels);

        fin_label.seekg(sizeof(unsigned char) * idx, std::ios::cur);

        data = 0;
        fin_label.read((char *)&data, sizeof(data));
        temp_label[(int)data] = 1.f;

        result->push_back(temp_label);
    }

    return result;
}


template<typename DTYPE> int MNIST<DTYPE>:: GetLength(){
    if(m_option == Train){
        return NUMOFTRAINDATA;
    } else if(m_option == Test){
        return NUMOFTESTDATA;
    }
}

template<typename DTYPE> int MNIST<DTYPE>::ReverseInt(DTYPE i) {
    unsigned char ch1, ch2, ch3, ch4;

    ch1 = (int)i & 255;
    ch2 = ((int)i >> 8) & 255;
    ch3 = ((int)i >> 16) & 255;
    ch4 = ((int)i >> 24) & 255;

    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
