#include "ImageNetReader.h"

#include <unistd.h>

int main(int argc, char const *argv[]) {
    std::cout << "======================START=========================" << '\n';

    ImageNetDataReader<float> *data_reader = new ImageNetDataReader<float>(200, 10, FALSE);

    Tensor<float> **data = NULL;
    data_reader->StartProduce();

    sleep(3);

    for(int i = 0; i < 250; i++){
        std::cout << "i : " << i << '\n';
        data = data_reader->GetDataFromBuffer();

        std::cout << data[1]->GetShape() << '\n';
        std::cout << data[0]->GetShape() << '\n';

        delete data[0];
        delete data[1];

        delete data;
    }

    data_reader->StopDataPreprocess();

    sleep(3);

    delete data_reader;

    std::cout << "======================Done=========================" << '\n';

    return 0;
}
