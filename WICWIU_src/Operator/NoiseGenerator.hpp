#ifndef NOISEGENERATOR_H_
#define NOISEGENERATOR_H_

#include "../Operator.hpp"

/*!
@class NoiseGenerator Class
@details Tensor 클래스의 Random_normal 함수를 사용하여 범위 내의 임의의 값을 갖는 Tensor 생성
@details Operator 형식이지만 Tensor를 저장하는 용도로만 사용
*/
template <typename DTYPE>
class NoiseGenerator : public Operator<DTYPE>
{
private:
public:
    NoiseGenerator(int pTimeSize, int pBatchSize, int pChannelSize, int pRowSize, int pColSize,
                   std::string pName = "No Name")
        : Operator<DTYPE>(pName)
    {
        this->SetResult(
            new Tensor<DTYPE>(pTimeSize, pBatchSize, pChannelSize, pRowSize, pColSize, NoUseTime));
    }

    NoiseGenerator(Shape* pShape, std::string pName = "No Name") : Operator<DTYPE>(pName)
    {
        this->SetResult(new Tensor<DTYPE>(pShape, NoUseTime));
    }

    ~NoiseGenerator() {}
};
#endif // NOISEGENERATOR_H_