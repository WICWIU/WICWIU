#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>

#include "../../WICWIU_src/Tensor.h"

using namespace std;

template<typename DTYPE> class ImageNetDataReader {
private:
    int Alloc() {
        // allocate shared memory space
        return TRUE;
    }

    void Delete() {
        // delete shared memory space
    }

public:
    ImageNetDataReader(int batchSize, int bufferSize = 10) {
        Alloc();
    }

    virtual ~ImageNetDataReader() {
        Delete();
    }

    int DataPreprocess() {
        // on thread
        // if buffer is full, it need to be sleep
        // When buffer has empty space again, it will be wake up
        return TRUE;
    }

    int CheckClassList() {
        // for label one-hot vector
        // fix class index
        ///mnt/ssd/Data/ImageNet/synset_words.txt - class
        return TRUE;
    }

    int txt2ArrayImageList() {
        // for image preprocessing
        // when we have a list of image, we can shuffle the set of image data
 
        //
        return TRUE;
    }

    Tensor<DTYPE>* Image2Tensor(  /*Address of Image*/) {
        // develop
        // library 사용

        // open image with file name
        return NULL;
    }

    Tensor<DTYPE>* Label2Tensor(  /*Address of Label*/) {
        // develop
        // library 사용
        return NULL;
    }

    Tensor<DTYPE>* ConcatenateImage(  /*array of ImageTensor*/) {
        // develop
        // Do not consider efficiency
        return NULL;
    }

    Tensor<DTYPE>* ConcatenateLabel(  /*array of LabelTensor*/) {
        // develop
        // library 사용
        return NULL;
    }

    int AddData2Buffer(Tensor<DTYPE> **imageAndLabel) {
        return TRUE;
    }

    Tensor<DTYPE>** GetDataFromBuffer() {
        // circular queue
        return NULL;
    }
};
