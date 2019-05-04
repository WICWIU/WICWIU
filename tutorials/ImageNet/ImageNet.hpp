#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <queue>
#include <semaphore.h>
#include <pthread.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifdef __TURBOJPEG__
# include <turbojpeg.h>
#endif  // ifdef __TURBOJPEG__

#include "../../WICWIU_src/DataLoader.hpp"

#define NUMBER_OF_CLASS               1000
#define NUMBER_OF_CHANNEL             3
#define LEGNTH_OF_WIDTH_AND_HEIGHT    224
#define CAPACITY_OF_PLANE             50176
#define CAPACITY_OF_IMAGE             150528

#define TRAIN_DATA_PATH               "ILSVRC2012_img_train256"
#define TEST_DATA_PATH                "ILSVRC2012_img_val256"

template<typename DTYPE>
class ImageNetDataset : public Dataset<DTYPE>{
private:
    int m_numOfImg;
    std::string m_rootPath;
    std::string m_dataPath;
    std::string m_option;

    // set of name of Class
    int m_useClasNum;
    std::vector<std::string> m_className;
    int m_numOfImage;
    std::vector<std::string> m_aImage;
    std::vector<int> m_aLable;


    void CheckClassList();
    void CreateImageListOfEachClass();

public:
    ImageNetDataset(std::string rootPath, std::string dataPath, int useClassNum) {
        m_numOfImg = 0;
        m_rootPath = rootPath;
        m_dataPath = dataPath;
        m_useClasNum = useClassNum;
        assert((useClassNum > 0)&&(useClassNum <= NUMBER_OF_CLASS));
        // m_option   = option;

        Alloc();
        CheckClassList();
        CreateImageListOfEachClass();
    }

    virtual ~ImageNetDataset() {
        Delete();
    }

    virtual void                                                            Alloc();

    virtual void                                                            Delete();

    virtual std::vector<Tensor<DTYPE> *>                                  * GetData(int idx);

    virtual int                                                             GetLength();
};

template<typename DTYPE> void ImageNetDataset<DTYPE                         >::Alloc() {}

template<typename DTYPE> void ImageNetDataset<DTYPE                         >::Delete() {}

template<typename DTYPE> std::vector<Tensor<DTYPE> *> *ImageNetDataset<DTYPE>::GetData(int idx) {
    std::vector<Tensor<DTYPE> *> *result = new std::vector<Tensor<DTYPE> *>(0, NULL);

    Tensor<DTYPE> *image = Tensor<DTYPE>::Zeros(1, 1, 1, 1, CAPACITY_OF_IMAGE);
    Tensor<DTYPE> *label = Tensor<DTYPE>::Zeros(1, 1, 1, 1, NUMBER_OF_CLASS);

    return result;
}

template<typename DTYPE> int ImageNetDataset<DTYPE>::GetLength() {
    return 0;
}

template<typename DTYPE> void ImageNetDataset<DTYPE>::CheckClassList() {
    // mnt/ssd/Data/ImageNet/synset_words.txt
    std::string filePath  = m_rootPath + "/synset_words.txt";
    const char *cstr = filePath.c_str();

    // std::cout << filePath << '\n';

    FILE *pFile = NULL;

    pFile = fopen(cstr, "r");

    if (pFile == NULL) {
        printf("file open fail\n");
        exit(-1);
    } else {
        char realValue[20];

        for (int i = 0; i < m_useClasNum; i++) {
            if (fscanf(pFile, "%s", realValue)) {
                m_className.push_back((std::string)realValue);
                // m_className[i] = realValue;
                std::cout << i << " : " << m_className[i] << '\n';

                while (fgetc(pFile) != '\n') ;
            } else {
                printf("there is something error\n");
                exit(-1);
            }
        }
    }

    fclose(pFile);
}

template<typename DTYPE> void ImageNetDataset<DTYPE>::CreateImageListOfEachClass() {

    // list file : 1st line - number of image, the others - image file name
    for (int classNum = 0; classNum < m_useClasNum; classNum++) {
        std::string filePath  = m_rootPath + '/' + m_dataPath + '/' + m_className[classNum] + "/list.txt"; // check with printf
        const char *cstr = filePath.c_str();

        FILE *pFile = NULL;
        pFile = fopen(cstr, "r");

        char realValue[100];
        int numOfImageOfClass = 0;

        if (pFile == NULL) {
            printf("file open fail\n");
            exit(-1);
        } else {
            if (fscanf(pFile, "%s", realValue)) { // first realValue is already readed above
                numOfImageOfClass = atoi(realValue);
                for (int imageNum = 0; imageNum < numOfImageOfClass; imageNum++) {
                    if (fscanf(pFile, "%s", realValue)) {
                        m_aImage.push_back((std::string)(m_className[classNum] + '/' + realValue));
                        m_aLable.push_back(classNum);
                        // std::cout << m_aImage.back() << " : " << m_aLable.back()  << " : " << m_aLable.size() << '\n';
                    } else {
                        printf("there is something error\n");
                        exit(-1);
                    }
                }
            } else {
                printf("there is something error\n");
                exit(-1);
            }
        }
        fclose(pFile);
    }

    m_numOfImage = m_aImage.size();
    assert(m_numOfImage > 0);
    // std::cout << "m_numOfImage : " << m_numOfImage << '\n';

}
