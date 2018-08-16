#include "CIFAR100Reader.h"

#include <unistd.h>

int main(int argc, char const *argv[]) {
    std::cout << "======================START=========================" << '\n';

    CIFAR100Reader<float> *data_reader = new CIFAR100Reader<float>(200, 10, TRUE);

    Tensor<float> **data = NULL;

    data_reader->StartProduce();

    sleep(3);

    data = data_reader->GetDataFromBuffer();

    std::cout << data[1]->GetShape() << '\n';
    std::cout << data[0]->GetShape() << '\n';

    delete data[0];
    delete data[1];

    delete data;

    sleep(3);

    data = data_reader->GetDataFromBuffer();

    std::cout << data[1]->GetShape() << '\n';
    std::cout << data[0]->GetShape() << '\n';

    delete data[0];
    delete data[1];

    delete data;

    sleep(3);

    data_reader->StopProduce();

    sleep(3);

    delete data_reader;

    std::cout << "======================Done=========================" << '\n';

    return 0;
}
