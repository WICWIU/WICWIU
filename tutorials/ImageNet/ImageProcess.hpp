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

class TransformForImageWrapper {
private:
    /* data */

public:
    TransformForImageWrapper();
    virtual ~TransformForImageWrapper();
};
