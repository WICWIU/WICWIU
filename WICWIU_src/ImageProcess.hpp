#ifndef __ImageProcess__
#define __ImageProcess__

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
#include <iterator>
#include <dirent.h>
#include <sys/types.h>

#include "./Shape.hpp"

class ImageWrapper
{
public:
    unsigned char *imgBuf;
    Shape *imgShape;

    ImageWrapper()
    {
        imgBuf = NULL;
        imgShape = NULL;
    }

    ~ImageWrapper()
    {
        if (imgBuf)
        {
            delete[] imgBuf;
            imgBuf = NULL;
        }

        if (imgShape)
        {
            delete imgShape;
            imgShape = NULL;
        }
    }
};

class Transform
{
private:
    /* data */

public:
    Transform()
    {
        // std::cout << "Transform" << '\n';
    }

    virtual ~Transform() {}

    virtual void DoTransform(ImageWrapper &imgWrp)
    {
        // std::cout << "do Transform" << '\n';
    }
};

class Compose : public Transform
{
private:
    std::vector<Transform *> m_listOfTransform;
    int m_size;

public:
    Compose(std::initializer_list<Transform *> lvalue) : m_listOfTransform(lvalue)
    {
        // std::cout << "Compose" << '\n';
        m_size = m_listOfTransform.size();
    }

    virtual ~Compose()
    {
        for (int i = 0; i < m_size; i++)
        {
            delete m_listOfTransform[i];
            m_listOfTransform[i] = NULL;
        }
    }

    virtual void DoTransform(ImageWrapper &imgWrp)
    {
        // std::cout << "do Compose" << '\n';
        // std::cout << "size: " << m_size << '\n';

        // std::cout << imgWrp.imgShape << '\n';
        for (int i = 0; i < m_size; i++)
        {
            // std::cout << "start i : " << i << '\n';
            m_listOfTransform[i]->DoTransform(imgWrp);
            // std::cout << imgWrp.imgShape << '\n';
            // std::cout << "end" << '\n';
        }
    }
};

#endif //     __ImageProcess__