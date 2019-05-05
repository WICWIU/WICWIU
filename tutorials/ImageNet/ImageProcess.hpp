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

    virtual void DoTransform(ImageWrapper& imgWrp) {}
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
        std::cout << "do Compose" << '\n';
        std::cout << "size: " << m_size << '\n';

        for (int i = 0; i < m_size; i++) {
            m_listOfTransform[i]->DoTransform(imgWrp);
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
        std::cout << "do CenterCrop" << '\n';
    }
};
}
