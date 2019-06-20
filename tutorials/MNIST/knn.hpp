#include <map>
#include <iterator>
#include <vector>
#include <utility>
#include "../../WICWIU_src/NeuralNetwork.hpp"
// #include <thread>

void calDist(int imgNum, Tensor<float> * pred, Tensor<float> * ref, std::multimap< float, int >& dist_map){
    int numOfRef = ref->GetBatchSize();
    int dimOfFeature = pred->GetColSize();

    Shape *predShape = pred->GetShape();
    Shape *refShape = ref->GetShape();

    for(int j = 0; j < numOfRef; j++){
        float distance = 0.f;
        for(int k = 0; k < dimOfFeature; k++){
            float temp = (*pred)[Index5D(predShape,0,0,imgNum,0,k)] - (*ref)[Index5D(refShape,0,0,j,0,k)];
            distance += temp * temp;
            // printf("pred: %f, ref: %f, temp: %f\n", (*pred)[Index5D(predShape,0,0,imgNum,0,k)], (*ref)[Index5D(refShape,0,0,j,0,k)], temp);
            // std::cin >> temp;
            // (*dist)[i * numOfRef + j] += temp * temp;
        }
        dist_map.insert(pair<float, int>(distance, j));
    }
}

int onehot2label(int imgNum, Tensor<float> * labelOfPred){
    int numOfClass = labelOfPred->GetColSize();
    for(int j = 0; j < numOfClass; j++){
        // std::cout << (*labelOfPred)[imgNum * numOfClass + j] << ' ';
        if((*labelOfPred)[imgNum * numOfClass + j] == 1) return j;
    }
    return -1;
}

int knn(int k, Tensor<float> * labelOfRef, std::multimap< float, int >& dist_map){
    // for debuging
    int numOfClass = labelOfRef->GetColSize();
    std::vector<int> v(numOfClass, 0);
    std::multimap< float, int >::iterator iter = dist_map.begin();
    for(int i = 0; i < k; ++iter, ++i){
        float imgNum = (iter)->second;
        // std::cout << (iter)->first << " " << imgNum << '\n';
        for(int j = 0; j < numOfClass; j++){
            v[j] += (*labelOfRef)[imgNum * numOfClass + j];
        }
    }

    // for(int j = 0; j < numOfClass; j++){
    //     std::cout << v[j] << ' ';
    // }
    // std::cout << '\n';

    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));;
}

float GetAccuracy(int k, Operator<float> * pred, Operator<float> * labelOfPred, Operator<float> * ref, Operator<float> * labelOfRef){
    Tensor<float> *pred_t = pred->GetResult();
    Tensor<float> *ref_t =  ref->GetResult();
    Tensor<float> *labelOfPred_t = labelOfPred->GetResult();
    Tensor<float> *labelOfRef_t = labelOfRef->GetResult();

    int numOfImg = pred_t->GetBatchSize();
    int predClass = 0;
    int realClass = 0;
    int correct = 0;

    // create strorage to save distance
    std::multimap< float, int > dist_map;

    // calculate all distance
    for(int i = 0; i < numOfImg; i++){
        calDist(i, pred_t, ref_t, dist_map);
        predClass = knn(k, labelOfRef_t, dist_map);
        realClass = onehot2label(i, labelOfPred_t);
        if(predClass == realClass) correct++;
        // std::cout << "predClass: " << predClass << " realClass: " << realClass << '\n';
        // std::cin >> predClass;
        dist_map.clear();
    }

    return (float)correct / numOfImg;
}
