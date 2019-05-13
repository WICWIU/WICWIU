#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <queue>
#include <semaphore.h>
#include <pthread.h>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "../../WICWIU_src/Shape.hpp"

class ImageWrapper {
public:
    unsigned char *imgBuf;
    Shape *imgShape;

    ~ImageWrapper() {
        if (imgBuf) {
            delete[] imgBuf;
            imgBuf = NULL;
        }

        if (imgShape) {
            delete imgShape;
            imgShape = NULL;
        }
    }
};

namespace vision {
class Transform {
private:
    /* data */

public:
    Transform() {
        // std::cout << "Transform" << '\n';
    }

    virtual ~Transform() {}

    virtual void DoTransform(ImageWrapper& imgWrp) {
        // std::cout << "do Transform" << '\n';
    }
};

class Compose : public Transform {
private:
    std::vector<Transform *> m_listOfTransform;
    int m_size;

public:
    Compose(std::initializer_list<Transform *> lvalue) : m_listOfTransform(lvalue) {
        // std::cout << "Compose" << '\n';
        m_size = m_listOfTransform.size();
    }

    virtual ~Compose() {
        for (int i = 0; i < m_size; i++) {
            delete m_listOfTransform[i];
            m_listOfTransform[i] = NULL;
        }
    }

    virtual void DoTransform(ImageWrapper& imgWrp) {
        // std::cout << "do Compose" << '\n';
        // std::cout << "size: " << m_size << '\n';

        // std::cout << imgWrp.imgShape << '\n';
        for (int i = 0; i < m_size; i++) {
            // std::cout << "start i : " << i << '\n';
            m_listOfTransform[i]->DoTransform(imgWrp);
            // std::cout << imgWrp.imgShape << '\n';
            // std::cout << "end" << '\n';
        }
    }
};

class CenterCrop : public Transform {
private:
    int m_heigth;
    int m_width;

public:
    CenterCrop(int heigth, int width) : m_heigth(heigth), m_width(width) {
        // std::cout << "CenterCrop" << '\n';
    }

    CenterCrop(int size) : m_heigth(size), m_width(size) {
        // std::cout << "CenterCrop" << '\n';
    }

    virtual ~CenterCrop() {}

    virtual void DoTransform(ImageWrapper& imgWrp) {
        unsigned char *imgBuf = imgWrp.imgBuf;
        Shape *imgShape       = imgWrp.imgShape;

        int width = imgShape->GetDim(2);

        // assert(width >= m_width);  // assert는 전처리 코드로 막을 것
        int height = imgShape->GetDim(1);
        // assert(height >= m_heigth);
        int channel = imgShape->GetDim(0);

        unsigned char *newImgBuf = new unsigned char[m_width * m_heigth * channel];
        Shape *NewImgShape       = new Shape(channel, m_heigth, m_width);

        int start_h = (height - m_heigth) / 2;
        int start_w = (width - m_width) / 2;

        for (int h = 0; h < m_heigth; h++) {
            int oldh = start_h + h;

            for (int w = 0; w < m_width; w++) {
                int oldw = start_w + w;

                for (int ch = 0; ch < channel; ch++) {
                    newImgBuf[h * m_width * channel + w * channel + ch]
                        = imgBuf[oldh * width * channel + oldw * channel + ch];
                }
            }
        }

        delete[] imgBuf;
        imgBuf = NULL;
        delete imgShape;
        imgShape = NULL;

        imgWrp.imgBuf   = newImgBuf;
        imgWrp.imgShape = NewImgShape;
    }
};

class RandomCrop : public Transform {
private:
    int m_heigth;
    int m_width;

public:
    RandomCrop(int heigth, int width) : m_heigth(heigth), m_width(width) {
    }

    RandomCrop(int size) : m_heigth(size), m_width(size) {
    }

    virtual ~RandomCrop() {}

    virtual void DoTransform(ImageWrapper& imgWrp) {
        unsigned char *imgBuf = imgWrp.imgBuf;
        Shape *imgShape       = imgWrp.imgShape;

        int width = imgShape->GetDim(2);
        int height = imgShape->GetDim(1);
        int channel = imgShape->GetDim(0);

        unsigned char *newImgBuf = new unsigned char[m_width * m_heigth * channel];
        Shape *NewImgShape       = new Shape(channel, m_heigth, m_width);
        srand((unsigned)time(NULL) + (unsigned int)(intptr_t)NewImgShape);

        int limit_h = height - m_heigth;
        int limit_w = width - m_width;

        int start_h = rand() % limit_h;
        int start_w = rand() % limit_w;

        for (int h = 0; h < m_heigth; h++) {
            int oldh = start_h + h;

            for (int w = 0; w < m_width; w++) {
                int oldw = start_w + w;

                for (int ch = 0; ch < channel; ch++) {
                    newImgBuf[h * m_width * channel + w * channel + ch]
                        = imgBuf[oldh * width * channel + oldw * channel + ch];
                }
            }
        }

        delete[] imgBuf;
        imgBuf = NULL;
        delete imgShape;
        imgShape = NULL;

        imgWrp.imgBuf   = newImgBuf;
        imgWrp.imgShape = NewImgShape;
    }
};

class HorizentalFlip : public Transform {
private:
    int m_heigth;
    int m_width;

public:
    HorizentalFlip(int heigth, int width) : m_heigth(heigth), m_width(width) {
    }

    HorizentalFlip(int size) : m_heigth(size), m_width(size) {
    }

    virtual ~HorizentalFlip() {}

    virtual void DoTransform(ImageWrapper& imgWrp) {
        unsigned char *imgBuf = imgWrp.imgBuf;
        Shape *imgShape       = imgWrp.imgShape;

        int width = imgShape->GetDim(2);
        int height = imgShape->GetDim(1);
        int channel = imgShape->GetDim(0);

        unsigned char *newImgBuf = new unsigned char[m_width * m_heigth * channel];
        Shape *NewImgShape       = new Shape(channel, m_heigth, m_width);


        //가로 플립.
        for (int h = 0; h < m_heigth; h++) {
            for (int w = 0; w < m_width; w++) {
                for (int ch = 0; ch < channel; ch++) {
                    newImgBuf[h * m_width * channel + w * channel + ch]
                        = imgBuf[h * width * channel + (width - w - 1) * channel + ch];
                }
            }
        }

        delete[] imgBuf;
        imgBuf = NULL;
        delete imgShape;
        imgShape = NULL;

        imgWrp.imgBuf   = newImgBuf;
        imgWrp.imgShape = NewImgShape;
    }
};

class VerticalFlip : public Transform {
private:
    int m_heigth;
    int m_width;

public:
    VerticalFlip(int heigth, int width) : m_heigth(heigth), m_width(width) {
    }

    VerticalFlip(int size) : m_heigth(size), m_width(size) {
    }

    virtual ~VerticalFlip() {}

    virtual void DoTransform(ImageWrapper& imgWrp) {
        unsigned char *imgBuf = imgWrp.imgBuf;
        Shape *imgShape       = imgWrp.imgShape;

        int width = imgShape->GetDim(2);
        int height = imgShape->GetDim(1);
        int channel = imgShape->GetDim(0);

        unsigned char *newImgBuf = new unsigned char[m_width * m_heigth * channel];
        Shape *NewImgShape       = new Shape(channel, m_heigth, m_width);


        //세로 플립.
        for (int h = 0; h < m_heigth; h++) {
            for (int w = 0; w < m_width; w++) {
                for (int ch = 0; ch < channel; ch++) {
                    newImgBuf[h * m_width * channel + w * channel + ch]
                        = imgBuf[(height - h -1) * width * channel + w * channel + ch];
                }
            }
        }

        delete[] imgBuf;
        imgBuf = NULL;
        delete imgShape;
        imgShape = NULL;

        imgWrp.imgBuf   = newImgBuf;
        imgWrp.imgShape = NewImgShape;
    }
};

class Resize : public Transform {
private:
    int newHeight;
    int newWidth;

public:
    Resize(int heigth, int width) : newHeight(heigth), newWidth(width) {
        // std::cout << "CenterCrop" << '\n';
    }

    Resize(int size) : newHeight(size), newWidth(size) {
        // std::cout << "CenterCrop" << '\n';
    }

    virtual ~Resize() {}

    virtual void DoTransform(ImageWrapper& imgWrp) {
        unsigned char *oldImgBuf = imgWrp.imgBuf;
        Shape *oldImgShape       = imgWrp.imgShape;

        int oldWidth  = oldImgShape->GetDim(2);
        int oldHeight = oldImgShape->GetDim(1);
        int channel   = oldImgShape->GetDim(0);

        unsigned char *newImgBuf = new unsigned char[newWidth * newHeight * channel];
        Shape *NewImgShape       = new Shape(channel, newHeight, newWidth);


        for (int newy = 0; newy < newHeight; newy++) {
            int oldy = newy * oldHeight / newHeight;

            for (int newx = 0; newx < newWidth; newx++) {
                int oldx = newx * oldWidth / newWidth;

                for (int c = 0; c < channel; c++) {
                    newImgBuf[newy * newWidth * channel + newx * channel + c]
                        = oldImgBuf[oldy * oldWidth * channel + oldx * channel + c];
                }
            }
        }

        delete[] oldImgBuf;
        oldImgBuf = NULL;
        delete oldImgShape;
        oldImgShape = NULL;

        imgWrp.imgBuf   = newImgBuf;
        imgWrp.imgShape = NewImgShape;
    }
};
}
