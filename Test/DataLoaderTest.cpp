#include "../WICWIU_src/NeuralNetwork.hpp"
#include "../WICWIU_src/DataLoader.hpp"

#include <unistd.h>

int main(int argc, char const *argv[]) {
    Dataset<float> *ds = new Dataset<float>();
    DataLoader<float> * dl = new DataLoader<float>(ds, 5, FALSE, 5, TRUE);

    sleep(2);

    delete ds;
    delete dl;

    return 0;
}
